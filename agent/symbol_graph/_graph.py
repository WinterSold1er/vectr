"""
SQLite-backed SymbolGraph: locate, trace, format, and persistence.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Literal

from agent.render_paths import workspace_relpath
from agent.symbol_graph._types import Symbol, LocateResult, CallEdge
from agent.symbol_graph._constants import (
    SNIPPET_LINES,
    SYMBOL_SCHEMA_VERSION,
    _KIND_RANK,
    _KIND_RANK_DEFAULT,
    _BUILTINS,
    _IMPORTANCE_SYMBOL_KINDS,
    _IMPL_CONTAINER_NODE_TYPES,
    graph_toolchain_fingerprint,
)
from agent.config import (
    LOCATE_LARGE_SPAN_THRESHOLD,
    LOCATE_SMALL_SPAN_THRESHOLD,
    INGEST_TRACES_MAX_UNRESOLVED_EXAMPLES,
    SEARCH_IDENTIFIER_HINT_NEARMISS_MIN_PREFIX_LEN,
    SYMBOL_GRAPH_TRACE_QUALIFIER_NEARMISS_MAX,
    SYMBOL_GRAPH_QUALIFIED_NEARMISS_CANDIDATE_CAP,
)

logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    """Compute edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _get_imported_files(caller_file: str, workspace: str) -> list[str]:
    """Parse caller_file's import statements and return workspace-resident file paths it imports."""
    workspace_path = Path(workspace)
    caller_path = Path(caller_file)
    try:
        src = caller_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    # Build stem → path index for workspace Python files
    name_index: dict[str, str] = {}
    for f in workspace_path.rglob("*.py"):
        name_index[f.stem] = str(f)
        try:
            rel = f.relative_to(workspace_path)
            key = str(rel).replace("/", ".").replace("\\", ".").removesuffix(f.suffix)
            name_index[key] = str(f)
        except ValueError:
            pass

    _PY_IMPORT = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)
    imported: list[str] = []
    for m in _PY_IMPORT.finditer(src):
        parts = m.group(1).split(".")
        for length in range(len(parts), 0, -1):
            candidate = ".".join(parts[:length])
            if candidate in name_index:
                resolved = name_index[candidate]
                if resolved != str(caller_path):
                    imported.append(resolved)
                break

    return list(dict.fromkeys(imported))  # dedup while preserving order


# UPG-CLASS-DEF-RE-MODIFIERS: a class declaration line may carry declaration
# modifiers before the `class` keyword — Java (`public`/`private`/`protected`/
# `final`/`abstract`/`static`/`sealed`/`strictfp`) and TS/JS (`export`/`default`/
# `declare`/`abstract`). The bare `^class` form missed exactly the majority
# forms (`public class`, `export class`, `export default class`), so the
# enclosing-class backward scan misread a modifier-prefixed class line as a
# scope-exit and dropped it — breaking qualified `Class.method` resolution on
# public/exported API surfaces. The prefix is a FIXED, language-syntax-anchored
# keyword set (definition-side parsing, not query-side classification); a
# modifier run that is not followed by `class` still fails to match, preserving
# the "non-class lesser-indent line exits the scope" semantics.
_CLASS_DEF_RE = re.compile(
    r"^(?:(?:export|default|public|private|protected|abstract|final|static|declare|sealed|strictfp)\s+)*"
    r"class\s+(\w+)"
)

# ARCH-2: whole-word identifier tokenizer used to count class-name reference
# frequency across the corpus (locate_scope-independent — a single pass over
# every file's text, not a per-class regex search).
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# UPG-CLASS-IMPORTANCE-REEXPORT-NOISE: a bare barrel / re-export line
# (`export { X } from '...'`, `export * from '...'`, `export * as NS from '...'`)
# is module plumbing, not a genuine reference to the named symbol — counting it
# inflates class_importance identically to a real call-site, enough that a
# single barrel re-export can double a symbol's reference count and invert
# display order against a large ce_relevance gap. Occurrences on a line matching
# this pattern are excluded from the reference-frequency count. A static,
# corpus-wide, index-time property of the occurrence's LINE (definition-side
# parsing, not query-side logic). Deliberately does NOT match `export class X`,
# `export const X`, or `export default X` — genuine definitions/values keep
# their count; and `import { X } from ...` is left counting (a real use site).
_REEXPORT_LINE_RE = re.compile(
    r"^\s*export\s+(?:\{[^}]*\}|\*(?:\s+as\s+\w+)?)\s+from\s+['\"]"
)

# UPG-15.10: detect enclosing function/method scopes for locate scope-depth ranking.
# Matches `def ` and `async def ` at the start of a stripped line so indented
# method defs inside a class are correctly identified as function scopes.
_DEF_LINE_RE = re.compile(r"^(def |async def )")


def _locate_scope_depth_from_lines(lines: list[str], start_line: int) -> int:
    """Count how many enclosing function/method (def) scopes surround the symbol
    at *start_line* (1-indexed) within a pre-read *lines* list.

    A top-level class definition has scope_depth=0; a class defined inside a test
    method body (e.g. ``def test_invalid_model(self): class Model(...): ...``) has
    scope_depth=1; a class inside a nested function has scope_depth=2; etc.

    Algorithm: walk backward from the symbol's line, collecting the set of distinct
    indentation levels at which a ``def``/``async def`` appears.  Each distinct
    indent level that is strictly less than the symbol's indent level and was not
    yet "consumed" by a shallower ancestor counts as one enclosing function scope.
    The fast path for indent=0 avoids any scanning (top-level symbols can never be
    inside a function).

    This function is called once per unique (file_path, start_line) pair per
    locate_l2 call via ``_locate_scope_depth_batch`` — not once per sort comparison.
    """
    if not lines or start_line < 1:
        return 0
    sym_idx = min(start_line - 1, len(lines) - 1)
    sym_line = lines[sym_idx]
    sym_indent = len(sym_line) - len(sym_line.lstrip())
    if sym_indent == 0:
        return 0  # top-level: fast path, cannot be inside any function

    # Collect distinct def-indentation levels that are strictly less than sym_indent
    # and appear before the symbol in the file.  Each distinct level represents one
    # enclosing function scope (e.g. a class method at indent=4 + a module-level
    # def at indent=0 → two distinct enclosing def levels → depth=2).
    enclosing_def_indents: set[int] = set()
    for idx in range(sym_idx - 1, -1, -1):
        line = lines[idx]
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if indent < sym_indent and _DEF_LINE_RE.match(stripped):
            enclosing_def_indents.add(indent)
    return len(enclosing_def_indents)


def _locate_scope_depth_batch(rows: list) -> list[int]:
    """Pre-compute scope depths for all rows in one pass, caching file reads.

    Returns a list of integers parallel to *rows*: ``depths[i]`` is the scope
    depth (number of enclosing function scopes) of ``rows[i]``.  Each source file
    is read at most once so the cost is O(unique_files × file_size), not
    O(rows × file_size).  Unreadable files silently return depth 0.
    """
    file_lines: dict[str, list[str]] = {}
    depths: list[int] = []
    for r in rows:
        fp = r["file_path"]
        if fp not in file_lines:
            try:
                file_lines[fp] = Path(fp).read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            except OSError:
                file_lines[fp] = []
        depths.append(_locate_scope_depth_from_lines(file_lines[fp], r["start_line"]))
    return depths


def _enclosing_class_from_lines(lines: list[str], start_line: int) -> str:
    """Scan backward from *start_line* (1-indexed) in a pre-read *lines* list to
    find the immediately enclosing class definition.

    Returns the class name string, or ``""`` when the symbol is not inside a
    class (e.g. a module-level function).  Only examines lines that are at a
    *lesser indentation level* than the symbol's first line, so nested methods
    are correctly attributed to their direct parent class rather than a grandparent.

    Split from ``_enclosing_class_from_file`` (UPG-15.10x/F49) so a batch caller
    can supply already-read lines once per file rather than re-reading it.
    """
    if not lines or start_line < 1:
        return ""

    # 0-indexed line of the symbol itself
    sym_idx = min(start_line - 1, len(lines) - 1)
    sym_line = lines[sym_idx]
    sym_indent = len(sym_line) - len(sym_line.lstrip())

    # Walk backward looking for a class definition at a strictly lesser indent
    for idx in range(sym_idx - 1, -1, -1):
        line = lines[idx]
        stripped = line.lstrip()
        if not stripped:
            continue  # blank line — keep scanning
        indent = len(line) - len(stripped)
        if indent >= sym_indent:
            continue  # same or deeper indent — not a parent scope
        m = _CLASS_DEF_RE.match(stripped)
        if m:
            return m.group(1)
        # Once we find a non-blank line at lesser indent that is NOT a class,
        # we've exited the potential enclosing class scope.
        break

    return ""


def _enclosing_class_from_file(file_path: str, start_line: int) -> str:
    """Scan *file_path* for the class immediately enclosing the symbol at
    *start_line* (1-indexed).  See ``_enclosing_class_from_lines`` for the
    scanning algorithm; this wrapper reads the file for one-off callers.

    This complements ``extract_class_from_content`` (which reads the
    ``# class: X`` prefix that the *indexer* injects into ChromaDB chunks).
    For the symbol-graph locate surface, snippets come directly from raw file
    content and carry no such prefix — hence the file-scan path here.
    """
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    return _enclosing_class_from_lines(lines, start_line)


def _impl_owner_type_name(node, language: str, code_bytes: bytes) -> str:
    """Owning type name of an impl-style container node (kind "impl" in
    `_SYMBOL_TYPES`), read from the grammar's own field rather than a
    positional child-scan.

    tree-sitter-rust exposes the implemented type as a `type` field on
    `impl_item` for BOTH `impl Type { }` and `impl Trait for Type { }` — the
    field always names the type after `for`, never the trait (`trait` is a
    separate, optional field present only on the second form), so this never
    misreads a trait name as the owner. A generic impl (`impl<T: Bound>
    Foo<T> { }`) exposes `type` as a `generic_type` node whose OWN `type`
    field is the bare `type_identifier` — unwrapped here down to the leaf.
    """
    if language != "rust":
        return ""
    cur = node.child_by_field_name("type")
    for _ in range(6):  # bounded — generic/reference nesting is shallow
        if cur is None:
            return ""
        if cur.type == "type_identifier":
            return code_bytes[cur.start_byte:cur.end_byte].decode("utf-8", errors="replace")
        cur = cur.child_by_field_name("type")
    return ""


def _impl_container_spans(file_path: str, language: str) -> list[tuple[int, int, str]]:
    """[(start_line, end_line, owner_type_name), ...] — 1-indexed inclusive —
    for every impl-style container block in *file_path* (UPG-RUST-IMPL-
    QUALIFIED-LOCATE). Computed once per file so a batch caller (below, and
    `_enclosing_container_from_file`) never re-parses the same file per row.

    This is the AST-based counterpart to the text/regex `class NAME` scan
    (`_enclosing_class_from_lines`) — needed because an impl-style container's
    header carries no single fixed keyword line to match: a Rust impl block's
    header can span multiple lines with generics (``impl<T: Clone>\n
    Foo<T>\n{``), and a trait impl names TWO types on one line
    (``impl Trait for Type {``) where only the second is the owner. Reading
    tree-sitter's own `type` field sidesteps both problems; a regex could only
    approximate the single-line, non-generic case.

    Returns [] when *language* has no impl-kind container (i.e. every
    language except Rust today — see `_IMPL_CONTAINER_NODE_TYPES`), its
    grammar isn't installed, or the file can't be read.
    """
    container_types = _IMPL_CONTAINER_NODE_TYPES.get(language, frozenset())
    if not container_types:
        return []
    from agent.indexer import _get_parser
    parser = _get_parser(language)
    if parser is None:
        return []
    try:
        code = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    spans: list[tuple[int, int, str]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in container_types:
            owner = _impl_owner_type_name(node, language, code_bytes)
            if owner:
                spans.append((node.start_point[0] + 1, node.end_point[0] + 1, owner))
        stack.extend(node.children)
    return spans


def _owner_from_spans(spans: list[tuple[int, int, str]], start_line: int) -> str:
    """Innermost (smallest-span) container from `_impl_container_spans` whose
    range contains *start_line* (1-indexed), or "" when none does."""
    best, best_size = "", None
    for s, e, owner in spans:
        if s <= start_line <= e and (best_size is None or (e - s) < best_size):
            best, best_size = owner, e - s
    return best


def _enclosing_container_from_file(file_path: str, start_line: int) -> str:
    """Enclosing definitional container (class OR impl-style block) for the
    symbol at *start_line* in *file_path* — the single "what type owns this
    member" lookup shared by qualified `Type.method` resolution in both
    `locate_l2` and `trace` (UPG-RUST-IMPL-QUALIFIED-LOCATE).

    Tries the fast text-based `class NAME` scan first (`_enclosing_class_from_file`
    — unchanged behaviour for every class-keyword language: Python, Java,
    JS/TS, C++). Falls back to the AST-based impl-container lookup only when
    that finds nothing — i.e. only for a language whose owning-container
    syntax isn't a `class`-keyword line at all. A Rust method's owner is
    declared by its enclosing `impl Type { }` block, never a `class` line, so
    the text scan structurally can never find it there; the fallback reads
    the owner directly off tree-sitter's grammar field instead of extending
    the regex with another keyword (a generic/multi-line impl header has no
    single line a keyword regex could match).
    """
    cls = _enclosing_class_from_file(file_path, start_line)
    if cls:
        return cls
    from agent.indexer import LANG_BY_EXT
    language = LANG_BY_EXT.get(Path(file_path).suffix.lower(), "")
    return _owner_from_spans(_impl_container_spans(file_path, language), start_line)


def _locate_class_enclosed_batch(rows: list) -> list[bool]:
    """Pre-compute, for each row, whether the symbol's IMMEDIATE enclosing scope
    is a class body OR an impl-style container (used by the F49 locate ranking
    signal below). Each source file is read at most once — mirrors
    ``_locate_scope_depth_batch``'s caching; the impl-container AST scan
    (UPG-RUST-IMPL-QUALIFIED-LOCATE) is likewise computed at most once per file.

    A method directly inside a class (e.g. ``class ForNode: def render(self):``)
    or inside an impl block (e.g. Rust ``impl Basket { fn total(&self) {} }``)
    returns True; a module-level function (e.g. ``def render(request, ...):``)
    returns False. A symbol whose immediate enclosing scope is a function body
    (e.g. a class defined inside a test method, per UPG-15.10/F29) also returns
    False here — that case is already penalised by ``scope_depth`` and
    ``test_penalty``; this signal targets the orthogonal "bare function vs.
    class method" collision that scope_depth alone does not discriminate,
    because Python's tree-sitter grammar assigns both the same symbol kind
    (`function`), and even languages with a distinct `method` kind rank it
    equal to `function` in `_KIND_RANK` (a method is a legitimate top-level
    result, e.g. `locate('save')` on a single canonical model).
    """
    from agent.indexer import LANG_BY_EXT
    file_lines: dict[str, list[str]] = {}
    impl_spans: dict[str, list[tuple[int, int, str]]] = {}
    result: list[bool] = []
    for r in rows:
        fp = r["file_path"]
        if fp not in file_lines:
            try:
                file_lines[fp] = Path(fp).read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            except OSError:
                file_lines[fp] = []
        enclosed = bool(_enclosing_class_from_lines(file_lines[fp], r["start_line"]))
        if not enclosed:
            if fp not in impl_spans:
                language = LANG_BY_EXT.get(Path(fp).suffix.lower(), "")
                impl_spans[fp] = _impl_container_spans(fp, language)
            enclosed = bool(_owner_from_spans(impl_spans[fp], r["start_line"]))
        result.append(enclosed)
    return result


def _enclosing_container_batch(symbols: list) -> list[str]:
    """Batch-cached counterpart to `_enclosing_container_from_file` for a
    list of `Symbol` objects — one file read (and, on the impl-container
    fallback path, one parse) per unique `file_path` across *symbols*,
    mirroring `_locate_class_enclosed_batch`'s per-file caching. Returns the
    enclosing class/impl-container name (or `""`) parallel to *symbols*.

    UPG-QUALIFIER-SIM-FILE-CACHE: `nearest_symbol_names`'s qualified
    near-miss ranking pool can hold up to
    `search.identifier_hint.qualified_nearmiss.candidate_pool_cap` candidates
    (default 50) — a common leaf name colliding across many same-file
    sibling classes/impl blocks previously meant `_qualifier_similarity_key`
    re-reading (and, on the impl-container path, re-parsing) that same file
    once per candidate defined in it.
    """
    from agent.indexer import LANG_BY_EXT
    file_lines: dict[str, list[str]] = {}
    impl_spans: dict[str, list[tuple[int, int, str]]] = {}
    result: list[str] = []
    for s in symbols:
        fp = s.file_path
        if fp not in file_lines:
            try:
                file_lines[fp] = Path(fp).read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            except OSError:
                file_lines[fp] = []
        enclosing = _enclosing_class_from_lines(file_lines[fp], s.start_line)
        if not enclosing:
            if fp not in impl_spans:
                language = LANG_BY_EXT.get(Path(fp).suffix.lower(), "")
                impl_spans[fp] = _impl_container_spans(fp, language)
            enclosing = _owner_from_spans(impl_spans[fp], s.start_line)
        result.append(enclosing)
    return result


# UPG-15.16: SQL pre-ranking order applied BEFORE the exact/suffix LIMIT cap so
# the canonical definition of a frequently-defined name enters the candidate pool
# that _partial_match_key then ranks. In any large codebase a common class or
# function name can have more exact matches than the cap — typically many
# test/fixture stubs plus one real library definition. Without ordering, SQLite
# returns an arbitrary rowid-ordered subset that can omit the real definition
# entirely, and that subset shifts across re-indexes as rowids change, so locate
# is both wrong and non-deterministic. Order non-test files first (tests/ dirs and
# test_*.py basenames), then larger line-span first (real defs tend to be large,
# stubs tiny); _partial_match_key does the precise ranking within the cap.
# (Witnessed on a large web-framework corpus: the name "Model" has 236 matches,
# mostly inner test classes, so the library definition fell outside a 200 cap.)
_CANONICAL_FETCH_ORDER = (
    "ORDER BY (CASE WHEN file_path LIKE '%/tests/%' "
    "OR file_path LIKE '%/test\\_%' ESCAPE '\\' THEN 1 ELSE 0 END), "
    "(end_line - start_line) DESC"
)


def _split_class_qualifier(name: str) -> tuple[str, str]:
    """Split a ``"Class.method"`` / ``"Class:method"`` query into
    ``(class_name, leaf)``. Returns ``("", name)`` — i.e. unqualified — when
    *name* has no ``.``/``:`` separator, or the left-hand side doesn't look
    like a class name (UpperCamelCase). This avoids misreading a dotted
    module path (``"os.path"``) as a class qualifier when the caller really
    wants ordinary suffix-stripping.

    Shared by ``locate_l2`` (UPG-11.10-b) and ``trace`` (F33): a vectr_search
    result displays symbols as ``"Repository.delete"`` (Class.method), and an
    LLM naturally copies that qualified name into a follow-up ``locate``/
    ``trace`` call — both call sites need the identical split+match
    convention so a qualified query resolves consistently across both tools.
    """
    if "." not in name and ":" not in name:
        return "", name
    parts = re.split(r"[.:]", name, maxsplit=1)
    if len(parts) == 2 and parts[0] and parts[1] and parts[0][0].isupper():
        return parts[0], parts[1]
    return "", name


def _partial_match_key(
    row, query_lower: str, scope_depth: int = 0, class_enclosed: bool = False,
) -> tuple:
    """Sort key for `locate` matches (used in every strategy via ``_ranked_result``).

    Ordering, most→least preferred (lower tuple = better rank):
      1. match position — exact (case-insensitive) > prefix > interior substring
      2. canonical kind — def > impl/alias (see _KIND_RANK)
      3. not a test/private file
      4. fewer enclosing function scopes — top-level (0) beats inner test class (1+)
         (UPG-15.10: discriminates inner test-scope stubs from canonical library defs)
      5. not enclosed in a class — a bare module-level definition beats a
         same-named class method (UPG-15.10x/F49: `shortcuts.render` — a short,
         module-level function — must beat `ForNode.render`, a same-named method
         whose larger body would otherwise win on span alone; placed BEFORE span
         so a small canonical function is never outranked by a bigger method
         look-alike). A no-op when every same-name candidate is a method (e.g.
         `locate('save')` across many model classes) — falls through to span_bucket.
      6. larger line span bucket — large (0) > medium (1) > tiny stub (2)
         (UPG-15.10: canonical 1400-line base class beats 3-line test stub)
      7. shorter name (closer to the query)
      8. file_path (stable tiebreak)
    """
    name = row["name"]
    nl = name.lower()
    if nl == query_lower:
        pos = 0
    elif nl.startswith(query_lower):
        pos = 1
    else:
        pos = 2
    kind_rank = _KIND_RANK.get(row["kind"], _KIND_RANK_DEFAULT)
    fp = row["file_path"]
    fp_low = fp.replace("\\", "/").lower()
    segments = fp_low.split("/")
    base = segments[-1]
    stem = base.rsplit(".", 1)[0]
    # Test-file / private-file detection must look at the basename and exact path
    # segments only — substring "test" in a path (e.g. pytest tmp dirs, a
    # "my_test_project" root) must NOT penalise an otherwise-canonical symbol.
    is_test = (
        stem.startswith(("test_", "test-")) or stem in ("test", "tests")
        or stem.endswith(("_test", ".test", ".spec", "_spec"))
        or any(seg in ("test", "tests", "testing", "__tests__") for seg in segments[:-1])
    )
    is_private = base.startswith("_")
    test_penalty = 1 if (is_test or is_private) else 0
    # UPG-15.10: line-span bucket — larger is more canonical.
    # ``row`` may be a sqlite3.Row (no .get()) or a plain dict; handle both.
    try:
        span = max(0, int(row["end_line"] or 0) - int(row["start_line"] or 0))
    except (KeyError, TypeError, IndexError):
        span = 0  # row missing line fields (synthetic test rows) → treat as tiny
    if span >= LOCATE_LARGE_SPAN_THRESHOLD:
        span_bucket = 0   # large canonical definition
    elif span >= LOCATE_SMALL_SPAN_THRESHOLD:
        span_bucket = 1   # medium
    else:
        span_bucket = 2   # tiny stub (inner test class, single-line function, etc.)
    # UPG-15.10: scope_depth — 0 = top-level, 1+ = nested inside function body.
    # Capped at 3 so deeply-nested test utilities don't dominate the key ordering.
    scope_penalty = min(scope_depth, 3)
    # UPG-15.10x/F49: class_enclosed — 0 = bare module-level def, 1 = class method.
    class_penalty = 1 if class_enclosed else 0
    return (pos, kind_rank, test_penalty, scope_penalty, class_penalty, span_bucket, len(name), fp)


# ---------------------------------------------------------------------------
# SQLite-backed SymbolGraph
# ---------------------------------------------------------------------------

class SymbolGraph:
    """
    Persistent symbol and call graph store.

    Answers:
      locate(name)  → which files define or reference a symbol
      callers(name) → which symbols call this one
      callees(name) → which symbols this one calls
    """

    def __init__(self, db_dir: str) -> None:
        self._db_path = Path(db_dir) / "symbol_graph.sqlite"
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace   TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    kind        TEXT NOT NULL,
                    file_path   TEXT NOT NULL,
                    start_line  INTEGER NOT NULL,
                    end_line    INTEGER NOT NULL,
                    description TEXT,
                    indexed_at  REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sym_workspace ON symbols(workspace);
                CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name);
                CREATE INDEX IF NOT EXISTS idx_sym_file ON symbols(file_path);

                CREATE TABLE IF NOT EXISTS edges (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace    TEXT NOT NULL,
                    from_file    TEXT NOT NULL,
                    from_symbol  TEXT NOT NULL,
                    from_line    INTEGER NOT NULL,
                    to_symbol    TEXT NOT NULL,
                    edge_type    TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_edge_workspace ON edges(workspace);
                CREATE INDEX IF NOT EXISTS idx_edge_from ON edges(from_symbol);
                CREATE INDEX IF NOT EXISTS idx_edge_to ON edges(to_symbol);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_unique
                    ON edges(workspace, from_file, from_symbol, from_line, to_symbol, edge_type);

                CREATE TABLE IF NOT EXISTS graph_meta (
                    workspace    TEXT NOT NULL,
                    key          TEXT NOT NULL,
                    value        TEXT,
                    PRIMARY KEY (workspace, key)
                );

                CREATE TABLE IF NOT EXISTS symbol_importance (
                    workspace   TEXT NOT NULL,
                    file_path   TEXT NOT NULL,
                    score       REAL NOT NULL,
                    PRIMARY KEY (workspace, file_path)
                );

                CREATE TABLE IF NOT EXISTS class_importance (
                    workspace       TEXT NOT NULL,
                    qualified_class TEXT NOT NULL,
                    score           REAL NOT NULL,
                    PRIMARY KEY (workspace, qualified_class)
                );

                CREATE TABLE IF NOT EXISTS file_fan_in (
                    workspace   TEXT NOT NULL,
                    file_path   TEXT NOT NULL,
                    count       INTEGER NOT NULL,
                    PRIMARY KEY (workspace, file_path)
                );
            """)

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------

    def index_file(self, workspace: str, file_path: str) -> int:
        """
        Index one file: extract symbols and call edges, store in DB.
        Returns number of symbols found.
        Replaces any previous index for this file.

        Calls extract_symbols_from_file through the package namespace so that
        test-time monkeypatching of agent.symbol_graph.extract_symbols_from_file
        is reflected here (identical to the original flat-module behaviour).
        """
        import agent.symbol_graph as _sg
        symbols, edges = _sg.extract_symbols_from_file(file_path)
        now = time.time()

        with self._conn() as conn:
            # delete previous entries for this file
            conn.execute(
                "DELETE FROM symbols WHERE workspace = ? AND file_path = ?",
                (workspace, file_path),
            )
            conn.execute(
                "DELETE FROM edges WHERE workspace = ? AND from_file = ?",
                (workspace, file_path),
            )

            for s in symbols:
                conn.execute(
                    """
                    INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (workspace, s["name"], s["kind"], file_path,
                     s["start_line"], s["end_line"], now),
                )

            for e in edges:
                conn.execute(
                    """
                    INSERT INTO edges (workspace, from_file, from_symbol, from_line, to_symbol, edge_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (workspace, e["from_file"], e["from_symbol"],
                     e["from_line"], e["to_symbol"], e["edge_type"]),
                )

        return len(symbols)

    def delete_file(self, workspace: str, file_path: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM symbols WHERE workspace = ? AND file_path = ?",
                (workspace, file_path),
            )
            conn.execute(
                "DELETE FROM edges WHERE workspace = ? AND from_file = ?",
                (workspace, file_path),
            )
            conn.execute(
                "DELETE FROM symbol_importance WHERE workspace = ? AND file_path = ?",
                (workspace, file_path),
            )
            # class_importance (ARCH-2) is keyed by class name, not file_path — a
            # class's reference count spans the whole corpus, not one file, so a
            # single-file delete can't correct it in isolation. It is recomputed
            # wholesale on the next build_for_workspace() (single-file index_file()
            # calls, used for incremental watcher updates, leave it stale until then).

    def build_for_workspace(
        self, workspace: str, file_paths: list[str], embed_model: str = "",
    ) -> dict:
        """
        Index all files in a workspace. Called after the main vector index is built.
        Returns {"symbols": int, "edges": int, "files": int, "failed": int, "complete": bool}

        Per-file resilient (UPG-8.7): a file that raises during extraction (e.g. a
        pathological AST that hits the recursion guard, an unreadable file) is
        skipped and counted — it can no longer abort the whole loop and silently
        leave every *later* file without symbols (the real cause of the observed
        "5531 symbols across 154 files" partial graph). After the build, the
        toolchain fingerprint + completeness are stamped so an upgrade is
        detectable and a partial build is never mistaken for a trustworthy one.
        """
        failed: list[str] = []
        for fp in file_paths:
            try:
                self.index_file(workspace, fp)
            except Exception:
                failed.append(fp)
                logger.warning("Symbol extraction failed for %s — skipped", fp, exc_info=True)

        # UPG-SYMBOLCOUNT-STALE: index_file() only DELETE+INSERTs rows for files
        # in *this* build's file_paths; rows from files a prior build indexed that
        # have since dropped out of the workspace (deleted / renamed / newly
        # excluded) linger in the warm-cache DB. Left in place they leak stale
        # symbols into locate/trace and inflate the status COUNT(*) above the
        # build's own symbol count (the observed +210 status-vs-build gap on a
        # warm-cache restart). Prune them so the persisted graph reflects exactly
        # the current file set, then derive the reported symbol count from the
        # SAME COUNT(*) query /v1/status serves — one source of truth, not two.
        current_files = set(file_paths)
        with self._conn() as conn:
            persisted_files = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT file_path FROM symbols WHERE workspace = ?", (workspace,)
                ).fetchall()
            ]
            for stale_fp in persisted_files:
                if stale_fp not in current_files:
                    conn.execute(
                        "DELETE FROM symbols WHERE workspace = ? AND file_path = ?",
                        (workspace, stale_fp),
                    )
                    conn.execute(
                        "DELETE FROM edges WHERE workspace = ? AND from_file = ?",
                        (workspace, stale_fp),
                    )
                    conn.execute(
                        "DELETE FROM symbol_importance WHERE workspace = ? AND file_path = ?",
                        (workspace, stale_fp),
                    )
            total_symbols = conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE workspace = ?", (workspace,)
            ).fetchone()[0]
            edge_count = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE workspace = ?", (workspace,)
            ).fetchone()[0]

        # ARCH-1a: compute file-level PageRank importance and persist it.
        self._compute_and_store_importance(workspace)

        # ARCH-2: compute class-level reference-frequency importance and persist it —
        # the signal that discriminates same-leaf method collisions (e.g. two classes
        # that both define a `delete` method) file-level importance cannot.
        self._compute_and_store_class_importance(workspace, file_paths)

        # UPG-TESTPATH-FRAMEWORK-MISCLASS (F58): compute corpus-wide unambiguous
        # caller-file fan-in and persist it — the signal that separates a shipped
        # testing-framework file living at a test-named path from disposable test
        # code at the same kind of path.
        self._compute_and_store_file_fan_in(workspace)

        if failed:
            logger.warning(
                "Symbol graph: %d/%d files failed extraction (e.g. %s) — graph is PARTIAL",
                len(failed), len(file_paths), ", ".join(Path(f).name for f in failed[:3]),
            )

        complete = not failed
        self._write_meta(workspace, {
            "fingerprint": graph_toolchain_fingerprint(embed_model),
            "schema_version": str(SYMBOL_SCHEMA_VERSION),
            "embed_model": embed_model,
            "files": str(len(file_paths)),
            "symbols": str(total_symbols),
            "failed": str(len(failed)),
            "complete": "1" if complete else "0",
            "built_at": str(time.time()),
        })

        # UPG-CALLER-INDEGREE-ANALYZE: a full workspace build bulk-inserts
        # thousands of rows through index_file()'s per-statement executes —
        # SQLite never gathers table/index statistics on its own after a
        # write, so the query planner keeps guessing off stale (or absent)
        # `sqlite_stat1` data. Without it, a selective lookup like
        # `_caller_in_degree`'s `to_symbol IN (...)` scan gets planned as a
        # full `idx_edge_unique` covering-index scan over the whole
        # workspace instead of a direct `idx_edge_to` index seek — ~15x
        # slower measured on a synthetic 300k-edge graph (~18.6ms →
        # ~1.0ms median for a 50-name lookup). The gather is an explicit
        # bounded `ANALYZE edges`, not `PRAGMA optimize`: optimize only
        # considers tables the *current connection* has read or written, and
        # `_conn()` hands out a brand new connection per call, so on the
        # SQLite builds that honour that rule the pragma is a silent no-op
        # here and `sqlite_stat1` is never created (observed as a Linux-only
        # CI failure against a macOS-passing local build). `analysis_limit`
        # caps the rows examined per index, which is SQLite's own recipe for
        # a few-millisecond ANALYZE on a large table.
        with self._conn() as conn:
            conn.execute("PRAGMA analysis_limit=400")
            conn.execute("ANALYZE edges")

        return {
            "symbols": total_symbols, "edges": edge_count, "files": len(file_paths),
            "failed": len(failed), "complete": complete,
        }

    # ------------------------------------------------------------------
    # Build metadata / version stamp (UPG-8.7)
    # ------------------------------------------------------------------

    def _write_meta(self, workspace: str, meta: dict[str, str]) -> None:
        with self._conn() as conn:
            for k, v in meta.items():
                conn.execute(
                    "INSERT INTO graph_meta (workspace, key, value) VALUES (?, ?, ?) "
                    "ON CONFLICT(workspace, key) DO UPDATE SET value = excluded.value",
                    (workspace, k, v),
                )

    def graph_meta(self, workspace: str) -> dict[str, str]:
        """Stored build stamp for this workspace ({} if never built)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value FROM graph_meta WHERE workspace = ?", (workspace,)
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def is_stale(self, workspace: str, embed_model: str = "") -> bool:
        """True if the persisted graph was built by a different toolchain
        (vectr upgrade / parser change / model change) or left incomplete, so a
        full rebuild is warranted. A never-built graph is stale. (UPG-8.7)"""
        meta = self.graph_meta(workspace)
        if not meta:
            return True
        if meta.get("complete") != "1":
            return True
        return meta.get("fingerprint") != graph_toolchain_fingerprint(embed_model)

    # ------------------------------------------------------------------
    # ARCH-1a: file-level PageRank importance
    # ------------------------------------------------------------------

    def _compute_and_store_importance(self, workspace: str) -> None:
        """Compute file-level PageRank over the def<->ref graph and persist scores.

        Algorithm (mirrors pagerank_spike.py):
          1. Build a leaf-name -> set(defining file_path) map from symbols.
          2. For each edge, look up the leaf of to_symbol; distribute weight
             1/|defs| to each defining file (skip self-edges).
          3. Run power-iteration PageRank (damping=0.85, 60 iterations).
          4. Normalize scores to [0,1] by dividing by the max score.
          5. Bulk-insert into symbol_importance (delete-first for idempotency).

        Pure stdlib — no numpy/networkx/scipy.
        """
        import collections

        with self._conn() as conn:
            # 1. leaf name -> set of defining file_paths
            name_to_files: dict[str, set[str]] = collections.defaultdict(set)
            for name, fp in conn.execute(
                "SELECT name, file_path FROM symbols WHERE workspace = ?", (workspace,)
            ):
                leaf = name.split(".")[-1]
                name_to_files[leaf].add(fp)

            # All distinct files that have at least one symbol
            files: set[str] = set(
                fp for (fp,) in conn.execute(
                    "SELECT DISTINCT file_path FROM symbols WHERE workspace = ?", (workspace,)
                )
            )

            if not files:
                # No symbols → nothing to compute; clear any stale rows.
                conn.execute(
                    "DELETE FROM symbol_importance WHERE workspace = ?", (workspace,)
                )
                return

            # 2. Build weighted file->file adjacency: out[from_file][to_file] += weight
            out: dict[str, dict[str, float]] = collections.defaultdict(
                lambda: collections.defaultdict(float)
            )
            for from_file, to_symbol in conn.execute(
                "SELECT from_file, to_symbol FROM edges WHERE workspace = ?", (workspace,)
            ):
                leaf = to_symbol.split(".")[-1]
                defs = name_to_files.get(leaf)
                if not defs:
                    continue
                w = 1.0 / len(defs)
                for df in defs:
                    if df == from_file:
                        continue  # skip self-edges
                    out[from_file][df] += w

        # 3. Power-iteration PageRank (damping=0.85, 60 iterations)
        nodes = list(files)
        N = len(nodes)
        d = 0.85

        # Pre-compute total out-weight per node
        outw: dict[str, float] = {f: sum(t.values()) for f, t in out.items()}

        # Initialize uniform PageRank
        pr: dict[str, float] = {f: 1.0 / N for f in nodes}

        for _ in range(60):
            new: dict[str, float] = {f: (1.0 - d) / N for f in nodes}

            # Dangling nodes: files with no outgoing edges contribute their PR
            # mass redistributed uniformly (standard dangling-node handling).
            dangling = 0.0
            for f in nodes:
                if outw.get(f, 0.0) == 0.0:
                    dangling += pr[f]
            dshare = d * dangling / N

            for f, targets in out.items():
                if f not in pr:
                    # from_file has no symbols (not in the nodes set) — skip
                    continue
                base = d * pr[f] / outw[f]
                for tf, w in targets.items():
                    if tf in new:
                        new[tf] += base * w

            for f in nodes:
                new[f] += dshare

            # Re-normalize to sum=1 to keep numerical stability
            s = sum(new.values())
            pr = {f: v / s for f, v in new.items()}

        # 4. Normalize to [0,1] by dividing by max score
        max_score = max(pr.values()) if pr else 0.0
        if max_score <= 0.0:
            # Degenerate: uniform graph or single node — store nothing meaningful
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM symbol_importance WHERE workspace = ?", (workspace,)
                )
            return

        normalized: dict[str, float] = {f: v / max_score for f, v in pr.items()}

        # 5. Persist: delete old rows first, then bulk-insert in one transaction.
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM symbol_importance WHERE workspace = ?", (workspace,)
            )
            conn.executemany(
                "INSERT INTO symbol_importance (workspace, file_path, score) VALUES (?, ?, ?)",
                ((workspace, fp, score) for fp, score in normalized.items()),
            )

        logger.debug(
            "ARCH-1a importance: workspace=%s files=%d max_raw=%.6f",
            workspace, N, max_score,
        )

    def file_importance(self, workspace: str) -> dict[str, float]:
        """Return {file_path: score} for all files in *workspace* where score is
        the normalized (0,1] file-level PageRank importance computed at index time.

        Returns an empty dict if importance has not been computed yet (e.g. the
        workspace was indexed before ARCH-1a, or contains no symbols).

        This is the read API consumed by ARCH-1b (searcher reranking) — do not
        wire into the searcher here; expose only.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT file_path, score FROM symbol_importance WHERE workspace = ?",
                (workspace,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # UPG-TESTPATH-FRAMEWORK-MISCLASS: unambiguous corpus-wide fan-in
    # ------------------------------------------------------------------

    def _compute_and_store_file_fan_in(self, workspace: str) -> None:
        """Compute, per file, the count of distinct calling files across
        UNAMBIGUOUS edges and persist it.

        The call-graph only records the bare/leaf symbol name at each edge
        (``obj.method()`` -> ``method``), so a common method name (``__init__``,
        ``setUp``) is shared by many unrelated definitions and dilutes any
        raw caller count with false attribution. Restricting to edges whose
        target leaf name resolves to EXACTLY ONE defining file corpus-wide
        isolates genuine widespread reuse of a uniquely-named API from that
        leaf-ambiguity noise — this is the same ``name_to_files`` map
        ``_compute_and_store_importance`` already builds, reused here rather
        than recomputed.

        This is a general structural signal (unambiguous caller breadth), not
        a path-based or framework-name-based one: it happens to separate a
        shipped testing-framework file from disposable test code at a
        test-named path (UPG-TESTPATH-FRAMEWORK-MISCLASS / F58) because a
        framework's public API is called unambiguously from many distinct
        files, while a throwaway test module is not called from anywhere.
        """
        import collections

        with self._conn() as conn:
            name_to_files: dict[str, set[str]] = collections.defaultdict(set)
            for name, fp in conn.execute(
                "SELECT name, file_path FROM symbols WHERE workspace = ?", (workspace,)
            ):
                leaf = name.split(".")[-1]
                name_to_files[leaf].add(fp)

            callers: dict[str, set[str]] = collections.defaultdict(set)
            for from_file, to_symbol in conn.execute(
                "SELECT from_file, to_symbol FROM edges WHERE workspace = ?", (workspace,)
            ):
                leaf = to_symbol.split(".")[-1]
                defs = name_to_files.get(leaf)
                if not defs or len(defs) != 1:
                    continue  # ambiguous leaf (shared by >1 file) — skip
                (target_file,) = defs
                if target_file == from_file:
                    continue  # skip self-edges
                callers[target_file].add(from_file)

            conn.execute(
                "DELETE FROM file_fan_in WHERE workspace = ?", (workspace,)
            )
            if callers:
                conn.executemany(
                    "INSERT INTO file_fan_in (workspace, file_path, count) VALUES (?, ?, ?)",
                    (
                        (workspace, fp, len(caller_files))
                        for fp, caller_files in callers.items()
                    ),
                )

    def file_fan_in(self, workspace: str) -> dict[str, int]:
        """Return {file_path: count} — corpus-wide unambiguous caller-file count
        computed at index time.

        Returns an empty dict if fan-in has not been computed yet (e.g. the
        workspace was indexed before UPG-TESTPATH-FRAMEWORK-MISCLASS, or
        contains no symbols). This is the read API consumed by the searcher's
        test_deprioritised exemption — do not wire into the searcher here;
        expose only.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT file_path, count FROM file_fan_in WHERE workspace = ?",
                (workspace,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # ARCH-2: class-level "canonical = most-referenced" importance
    # ------------------------------------------------------------------

    def _compute_and_store_class_importance(
        self, workspace: str, file_paths: list[str],
    ) -> None:
        """Compute symbol-name reference-frequency importance and persist scores.

        Extends ARCH-1a's file-level importance to class/type/function-name
        granularity — the signal that discriminates same-leaf collisions (two
        unrelated classes that both define a `delete` method; a module-level
        function crowded by a same-file, rarely-referenced sibling; a base
        type's own definition chunk crowded by its narrower same-file variants,
        UPG-SIBLING-TYPEDEF-CROWDING) which file-level importance cannot: in a
        module with one dominant symbol, that symbol *is* the file, so
        file-level PageRank can't separate the canonical symbol's chunk from a
        same-file sibling's, and it can't separate two candidates that live in
        different but equally-"important" files at all. Call edges are leaf-only
        (no receiver-type resolution — `obj.delete()` edges only ever record the
        bare `delete` target), so no edge-based signal can attribute a call to
        one owner over another; whole-word reference frequency of the symbol's
        own NAME sidesteps that entirely (validated design — receiver-type
        resolution was evaluated and rejected as the lever; see docs/tasks.md
        ARCH-2).

        Algorithm:
          1. Collect the distinct names of every TYPE DEFINITION (class/struct/
             enum/interface/typedef — see `_IMPORTANCE_SYMBOL_KINDS`) plus every
             module-level FUNCTION defined in this workspace (symbols table).
             A type-def or module-level-function chunk is attributed to its OWN
             name at query time (searcher-side); a method chunk keeps the
             existing owning-class attribution — see (3) below.
          2. Single pass per file: tokenize into whole-word identifiers and count
             occurrences of the known names (definitions, instantiations, type
             hints, inheritance, calls, docs — anywhere the name appears as a
             whole word). O(corpus size), not O(names x corpus size).
          3. Normalize to (0,1] with `log(1 + count) / log(1 + max_count)`
             rather than a linear `count / max_count`. Once function names
             (which skew toward far higher raw counts than class names — a
             widely-called helper can dwarf every class's reference count) share
             the same map, a linear scale lets one high-frequency function name
             stretch `max_count` enough to compress every OTHER name's score
             toward the same near-zero value, erasing the very separation this
             table exists to preserve; log-scaling compresses the top of the
             range instead, so a decisive raw-count gap (e.g. 439 vs 43) still
             yields a visibly different score after normalization even when the
             map's max count grows by an order of magnitude. The multiplicative
             relevance gate (`1 + lambda * score`, lambda <= 1) bounds the
             factor to `[1, 1+lambda]` regardless of scale choice, so an adverse
             case (a less-central name outscoring the true answer, e.g. 34 vs
             85) still cannot overturn a clear cross-encoder relevance lead —
             log-scaling only affects whether decisive cases stay visibly
             separated, not the no-overturn guarantee.
          4. Bulk-insert into class_importance (delete-first for idempotency).

        A name is intentionally NOT qualified by defining file — same-named
        symbols across files (e.g. a repeated `Meta`/`Config` inner class) share
        one corpus-wide reference count, matching "canonical = most-referenced"
        rather than a per-definition-site count.

        The table/column names (`class_importance`, `qualified_class`) predate
        this extension and are kept as-is to avoid schema churn — the stored
        score is symbol-name importance generally, not class-only.

        Pure stdlib — no new dependency, mirrors _compute_and_store_importance.
        """
        with self._conn() as conn:
            placeholders = ",".join("?" * len(_IMPORTANCE_SYMBOL_KINDS))
            names: set[str] = {
                name for (name,) in conn.execute(
                    "SELECT DISTINCT name FROM symbols WHERE workspace = ? "
                    f"AND kind IN ({placeholders})",
                    (workspace, *_IMPORTANCE_SYMBOL_KINDS),
                )
            }

        if not names:
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM class_importance WHERE workspace = ?", (workspace,)
                )
            return

        counts: dict[str, int] = dict.fromkeys(names, 0)
        for fp in file_paths:
            try:
                text = Path(fp).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # UPG-CLASS-IMPORTANCE-REEXPORT-NOISE: scan per line so a bare
            # barrel re-export line's mentions can be excluded — they are
            # plumbing, not genuine references. Token counting is otherwise
            # identical to a whole-file _WORD_RE pass (identifiers never span
            # newlines).
            for line in text.splitlines():
                if _REEXPORT_LINE_RE.match(line):
                    continue
                for m in _WORD_RE.finditer(line):
                    tok = m.group(0)
                    if tok in counts:
                        counts[tok] += 1

        max_count = max(counts.values()) if counts else 0

        with self._conn() as conn:
            conn.execute(
                "DELETE FROM class_importance WHERE workspace = ?", (workspace,)
            )
            if max_count > 0:
                log_max = math.log1p(max_count)
                conn.executemany(
                    "INSERT INTO class_importance (workspace, qualified_class, score) "
                    "VALUES (?, ?, ?)",
                    (
                        (workspace, name, math.log1p(count) / log_max)
                        for name, count in counts.items() if count > 0
                    ),
                )

        logger.debug(
            "ARCH-2 symbol-name importance: workspace=%s names=%d max_raw=%d",
            workspace, len(names), max_count,
        )

    def class_importance(self, workspace: str) -> dict[str, float]:
        """Return {class_name: score} for all classes in *workspace* where score is
        the normalized (0,1] whole-word reference-frequency importance computed at
        index time (ARCH-2).

        Returns an empty dict if importance has not been computed yet (e.g. the
        workspace was indexed before ARCH-2, or contains no classes).

        This is the read API consumed by the searcher's ranking blend — do not
        wire into the searcher here; expose only.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT qualified_class, score FROM class_importance WHERE workspace = ?",
                (workspace,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def symbol_count(self, workspace: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE workspace = ?", (workspace,)
            ).fetchone()[0]

    def ingest_trace_data(
        self,
        workspace: str,
        trace_events: list[dict],
    ) -> dict:
        """Ingest runtime trace events and add dynamic call edges to the graph.

        Accepts a list of trace event dicts. Recognised fields:
          caller      — name of the calling function/symbol (required)
          callee      — name of the called function/symbol (required)
          caller_file — source file of the caller (optional, empty string if unknown)
          caller_line — line number of the call site (optional, 0 if unknown)

        Dynamic edges use edge_type="dynamic" to distinguish them from static
        analysis edges. This bridges the dynamic dispatch gap: calls via
        __getattr__, decorators, dependency injection, etc. that static analysis
        misses are captured here.

        A caller or callee that doesn't match any symbol the static extractor
        found in this workspace (UPG-7.3) is common and NOT an error — the
        target may be an external library, a runtime-only name, or a symbol
        kind the extractor doesn't capture — so the edge is still ingested.
        It is, however, reported back as `unresolved_callers` / `unresolved_callees`
        (with a capped list of example pairs) so the caller can tell a typo'd
        trace event apart from a genuinely dynamic edge.

        Returns {"ingested": int, "skipped_invalid": int, "unresolved_callers": int,
        "unresolved_callees": int, "unresolved_examples": list[str]}.
        """
        ingested = 0
        skipped = 0
        valid_events: list[tuple[str, str, str, int]] = []

        for ev in trace_events:
            caller = str(ev.get("caller", "")).strip()
            callee = str(ev.get("callee", "")).strip()
            if not caller or not callee:
                skipped += 1
                continue
            caller_file = str(ev.get("caller_file", "")).strip()
            caller_line = int(ev.get("caller_line", 0))
            valid_events.append((caller, callee, caller_file, caller_line))

        names = {n for pair in valid_events for n in (pair[0], pair[1])}
        known = self._known_symbol_names(workspace, list(names))

        unresolved_callers = 0
        unresolved_callees = 0
        unresolved_examples: list[str] = []

        with self._conn() as conn:
            for caller, callee, caller_file, caller_line in valid_events:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO edges
                        (workspace, from_file, from_symbol, from_line, to_symbol, edge_type)
                    VALUES (?, ?, ?, ?, ?, 'dynamic')
                    """,
                    (workspace, caller_file, caller, caller_line, callee),
                )
                ingested += 1

                caller_unknown = caller not in known
                callee_unknown = callee not in known
                if caller_unknown:
                    unresolved_callers += 1
                if callee_unknown:
                    unresolved_callees += 1
                if (caller_unknown or callee_unknown) and (
                    len(unresolved_examples) < INGEST_TRACES_MAX_UNRESOLVED_EXAMPLES
                ):
                    reason = (
                        "both unresolved" if caller_unknown and callee_unknown
                        else "caller unresolved" if caller_unknown
                        else "callee unresolved"
                    )
                    unresolved_examples.append(f"{caller} -> {callee} ({reason})")

        logger.info(
            "ingest_traces: %d edges added, %d skipped, %d unresolved callers, "
            "%d unresolved callees (workspace=%s)",
            ingested, skipped, unresolved_callers, unresolved_callees, workspace,
        )
        return {
            "ingested": ingested,
            "skipped_invalid": skipped,
            "unresolved_callers": unresolved_callers,
            "unresolved_callees": unresolved_callees,
            "unresolved_examples": unresolved_examples,
        }

    # ------------------------------------------------------------------
    # Query: locate
    # ------------------------------------------------------------------

    def locate(self, workspace: str, name: str, limit: int = 10) -> list[Symbol]:
        """
        Find where a symbol is defined. Supports partial match.
        Returns definition sites (start_line of the defining node).
        """
        sql = """
            SELECT * FROM symbols
            WHERE workspace = ? AND name LIKE ?
            ORDER BY
                CASE WHEN name = ? THEN 0 ELSE 1 END,
                length(name),
                file_path
            LIMIT ?
        """
        pattern = f"%{name}%"
        with self._conn() as conn:
            rows = conn.execute(sql, (workspace, pattern, name, limit)).fetchall()
        symbols = [self._row_to_symbol(r) for r in rows]
        for sym in symbols:
            sym.snippet = self.get_snippet(sym.file_path, sym.start_line, sym.end_line)
        logger.debug("locate '%s': %d results", name, len(symbols))
        return symbols

    def locate_l2(
        self,
        workspace: str,
        name: str,
        limit: int = 10,
        caller_file: str | None = None,
        with_near_miss: bool = False,
    ) -> LocateResult:
        """
        Multi-strategy L2 call resolution. Falls back through 5 strategies
        when exact name match fails.

        Strategies tried in order:
          0 exact       — name = ?
          1 suffix      — strip qualifier prefix (module.Foo → Foo)
          2 same_module — symbols in same directory as caller_file
          3 import_chain— symbols in files imported by caller_file
          4 substring   — name contained as substring, ranked canonical-first
                          (prefix > interior, def > impl/alias) — UPG-4.5
          5 fuzzy       — edit distance ≤ length-scaled threshold, last resort

        UPG-11.10-b: when *name* contains a class qualifier (``"Class.method"``
        form), strategies 1–5 use the bare leaf for the DB lookup and then
        filter the results to symbols whose enclosing class matches the
        qualifier.  The qualified ``Class.method`` form is also populated on
        the returned symbol's ``name`` field whenever the symbol lives inside
        a class (parity with the searcher's qualified display).  Bare-leaf
        queries (no ``.``) are unchanged.

        UPG-LOCATE-NEARMISS-WIRE-PRIMARY-TOOL: with ``with_near_miss=True``
        (the primary locate surface opts in; internal callers like the
        search-hint paths keep the default False so every total miss does not
        pay for suggestions it never renders), a "none" resolution also
        computes the deterministic near-miss candidates — the exact machinery
        of `nearest_symbol_names`, via `_near_miss_candidates` — and stores
        them on the result's `near_miss` field, leaving `symbols` empty so a
        miss can never masquerade as a match.
        """
        # UPG-11.10-b: detect "Class.method" qualifier in the query (shared
        # split convention with `trace` — see `_split_class_qualifier`).
        _class_qualifier, _leaf = _split_class_qualifier(name)

        def _with_snippets(rows: list) -> list[Symbol]:
            syms = [self._row_to_symbol(r) for r in rows]
            for s in syms:
                s.snippet = self.get_snippet(s.file_path, s.start_line, s.end_line)
                # UPG-11.10-b: populate qualified name on the symbol so callers
                # see "Class.method" rather than bare "method" — parity with
                # the searcher's qualified display via extract_class_from_content.
                # UPG-RUST-IMPL-QUALIFIED-LOCATE: the owning container may be a
                # class OR an impl-style block (Rust) — `_enclosing_container_from_file`
                # covers both.
                if "." not in s.name and "::" not in s.name:
                    cls = _enclosing_container_from_file(s.file_path, s.start_line)
                    if cls:
                        s.name = f"{cls}.{s.name}"
            return syms

        def _filter_by_class(syms: list[Symbol]) -> list[Symbol]:
            """Keep only symbols whose enclosing class matches *_class_qualifier*.
            Returns *syms* unmodified when no qualifier is active."""
            if not _class_qualifier:
                return syms
            return [
                s for s in syms
                # After _with_snippets, qualified symbols have "Class.method"
                if s.name.split(".")[0] == _class_qualifier
                or s.name.split("::")[0] == _class_qualifier
            ]

        name_lower = name.lower()
        leaf_lower = _leaf.lower()

        def _ranked_result(rows: list, strategy: str) -> LocateResult:
            # Canonical-first ordering (UPG-4.5 + UPG-15.10 + UPG-15.10x/F49): even
            # within an exact-name hit, lead with the type/fn definition and bury
            # impl blocks / aliases / class-method look-alikes that share the name.
            # Pre-compute scope depths and class-enclosure (each a file read, cached
            # per file) so the sort key includes both signals without re-reading files
            # per comparison.
            scope_depths = _locate_scope_depth_batch(rows)
            class_enclosed = _locate_class_enclosed_batch(rows)
            ranked = sorted(
                range(len(rows)),
                key=lambda i: _partial_match_key(
                    rows[i], leaf_lower, scope_depths[i], class_enclosed[i],
                ),
            )
            top_rows = [rows[i] for i in ranked[:limit]]
            syms = _with_snippets(top_rows)
            filtered = _filter_by_class(syms)
            # When the qualifier is active and all results were filtered out,
            # fall through (return None so the caller tries the next strategy).
            if _class_qualifier and not filtered:
                return None  # type: ignore[return-value]
            return LocateResult(
                symbols=filtered if _class_qualifier else syms,
                resolution_strategy=strategy,
                query=name,
            )

        # Strategy 0: exact name match (uses _leaf when qualifier is active)
        lookup_name = _leaf if _class_qualifier else name
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE workspace = ? AND name = ? "
                + _CANONICAL_FETCH_ORDER + " LIMIT ?",
                (workspace, lookup_name, 200),
            ).fetchall()
        if rows:
            result = _ranked_result(rows, "exact")
            if result is not None:
                return result

        # Strategy 1: suffix match — strip qualifier prefix (e.g. "module.Foo" → "Foo").
        # When a class qualifier was already parsed above, strategy 1 is skipped
        # (the leaf is already extracted) to avoid double-stripping.
        if not _class_qualifier:
            suffix = name
            for sep in (":", "."):
                if sep in name:
                    suffix = name.rsplit(sep, 1)[-1]
                    break
            if suffix != name:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT * FROM symbols WHERE workspace = ? AND name = ? "
                        + _CANONICAL_FETCH_ORDER + " LIMIT ?",
                        (workspace, suffix, 200),
                    ).fetchall()
                if rows:
                    result = _ranked_result(rows, "suffix")
                    if result is not None:
                        return result

        # Strategy 2: same-module — symbols in the same directory as caller_file
        if caller_file:
            caller_dir = str(Path(caller_file).parent)
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM symbols WHERE workspace = ? AND name LIKE ? "
                    "AND file_path LIKE ? LIMIT ?",
                    (workspace, f"%{lookup_name}%", f"{caller_dir}/%", 200),
                ).fetchall()
            if rows:
                result = _ranked_result(rows, "same_module")
                if result is not None:
                    return result

        # Strategy 3: import-chain — symbols in files imported by caller_file
        if caller_file:
            imported = _get_imported_files(caller_file, workspace)
            if imported:
                ph = ", ".join("?" * len(imported))
                with self._conn() as conn:
                    rows = conn.execute(
                        f"SELECT * FROM symbols WHERE workspace = ? AND name LIKE ? "
                        f"AND file_path IN ({ph}) LIMIT ?",
                        (workspace, f"%{lookup_name}%", *imported, 200),
                    ).fetchall()
                if rows:
                    result = _ranked_result(rows, "import_chain")
                    if result is not None:
                        return result

        # Strategy 4: substring — any symbol whose name contains the query. Always
        # fires when there's at least one match, so fuzzy is a true last resort.
        # Prefix matches lead interior ones, and canonical defs lead impls/aliases
        # (UPG-4.5: `rand` → randint/randfraction before any fuzzy junk). SQL surfaces
        # prefix matches first into the fetch cap; _partial_match_key does the rest.
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE workspace = ? AND name LIKE ? "
                "ORDER BY (CASE WHEN name LIKE ? THEN 0 ELSE 1 END), length(name) LIMIT ?",
                (workspace, f"%{lookup_name}%", f"{lookup_name}%", 200),
            ).fetchall()
        if rows:
            result = _ranked_result(rows, "substring")
            if result is not None:
                return result

        # Strategy 5: fuzzy — edit distance within a length-scaled threshold, and
        # only against names that share the first character. Short queries get a
        # tighter budget so `rand` (len 4) can't match `nan`/`add` (UPG-4.5).
        max_dist = 1 if len(lookup_name) <= 4 else 2
        first = leaf_lower[0] if leaf_lower else ""
        with self._conn() as conn:
            all_rows = conn.execute(
                "SELECT * FROM symbols WHERE workspace = ? AND ABS(LENGTH(name) - ?) <= ?",
                (workspace, len(lookup_name), max_dist),
            ).fetchall()
        fuzzy = [
            r for r in all_rows
            if r["name"] and r["name"][0].lower() == first
            and _levenshtein(leaf_lower, r["name"].lower()) <= max_dist
        ]
        if fuzzy:
            fuzzy.sort(key=lambda r: (_levenshtein(leaf_lower, r["name"].lower()), len(r["name"])))
            syms = _with_snippets(fuzzy[:limit])
            filtered = _filter_by_class(syms)
            if filtered or not _class_qualifier:
                return LocateResult(
                    symbols=filtered if _class_qualifier else syms,
                    resolution_strategy="fuzzy",
                    query=name,
                )

        # UPG-LOCATE-NEARMISS-WIRE-PRIMARY-TOOL: a total miss carries its
        # deterministic near-miss candidates on the result (opt-in — see the
        # docstring) so every consumer of the result object, not just the LLM
        # formatter, can surface them. `symbols` stays empty: these are
        # suggestions, never a resolution.
        return LocateResult(
            symbols=[],
            resolution_strategy="none",
            query=name,
            near_miss=(
                self._near_miss_candidates(workspace, name, limit)
                if with_near_miss else []
            ),
        )

    def nearest_symbol_names(self, workspace: str, token: str, limit: int) -> list[Symbol]:
        """Cheap, deterministic near-miss lookup for a token that already
        failed EXACT resolution (see `app.service.identifier_hint_nearmiss`,
        UPG-NEARMISS-SYMBOL-NAMES). Never returns an "exact" result — every
        candidate here is inexact by construction, and the caller is
        responsible for labeling it as a near-miss, never as a match.

        Reuses `locate_l2`'s own non-exact resolution strategies (suffix,
        same_module, import_chain, substring, fuzzy) first. Those already
        cover a token that is a typo of, or shares a substring with, a real
        symbol name — but they only compare in the direction "the SYMBOL
        name contains/resembles the token". The other common near-miss shape
        — the token itself IS a real, shorter symbol name with an extra
        trailing word misremembered onto it (e.g. token "CacheControlHeader"
        against the real symbol "CacheControl") — falls through all of
        `locate_l2`'s strategies to "none", because none of them checks
        whether the TOKEN contains a shorter existing name as a prefix. One
        additional bounded prefix-containment query against the same
        `symbols` table covers that shape without any new fuzzy/edit-distance
        machinery: an existing symbol name is a candidate only when the token
        literally starts with it and the name is at least
        `search.identifier_hint.nearmiss_min_prefix_len` characters long
        (excludes short, generic names from matching as a "prefix" of
        unrelated tokens).

        UPG-LOCATE-FALLBACK-NO-SIMILARITY: when `token` carries a
        "Class.method" qualifier, EVERY one of `locate_l2`'s strategies
        filters candidates down to definitions actually enclosed by that
        class (`_filter_by_class`) — so a class that matches nothing in the
        workspace always ends in "none", regardless of how many OTHER
        classes define the same leaf method. This near-miss lookup is
        explicitly inexact by construction, so it is the sanctioned place to
        widen out to the leaf's real definitions across every OTHER class
        and rank them by how closely each candidate's own enclosing
        type/path resembles the qualifier that failed to resolve — a
        fabricated or misremembered class name is usually a near-miss of a
        real one. Ranking against `class_qualifier` (the explicit argument
        the caller passed) is a result-side signal, not query-content
        gating.

        UPG-QUALIFIED-TYPO-NEARMISS: the LEAF segment can carry a typo too
        (e.g. "QuerySt.get_or_creat" — both segments misspelled), in which
        case even the exact-leaf pool above is empty and there is nothing to
        rank. Falls back to resolving the bare leaf alone through
        `locate_l2` — the exact same per-segment partial-match machinery a
        bare (unqualified) token already gets, not a new parallel fuzzy
        path — and ranks whatever it finds by the same qualifier-similarity
        key. A doubly-typo'd qualified token then still surfaces its correct
        symbol as a labeled-inexact suggestion instead of a zero-suggestion
        miss.

        UPG-LOCATE-NEARMISS-WIRE-PRIMARY-TOOL: the fallback stages documented
        above now live in `_near_miss_candidates`, and `locate_l2`'s "none"
        branch computes them itself (opt-in via `with_near_miss`) so the
        primary locate surface gets the same suggestions onto its RESULT
        object. This method delegates to that computation instead of
        duplicating it — one source of truth for what "nearest" means.
        """
        if limit <= 0:
            return []
        result = self.locate_l2(workspace, token, limit=limit, with_near_miss=True)
        if result.resolution_strategy not in ("exact", "none"):
            return result.symbols[:limit]
        if result.resolution_strategy == "none":
            # The "none" branch above already computed exactly this fallback
            # (qualifier widening + prefix containment) — reuse it rather than
            # re-running the same queries here.
            return result.near_miss[:limit]
        # Exact resolution: `token` resolved cleanly, so by this method's own
        # contract ("a token that already failed EXACT resolution") there is
        # no near-miss to offer — never widen a correct answer into
        # suggestions (UPG-NEARMISS-WIDEN-EXACT-GUARD's principle, applied to
        # the bare-name case). No production caller reaches this with an
        # exact-resolved token anyway: identifier_hint_nearmiss guards exact
        # tokens out before calling, and trace() calls only for an unresolved
        # qualifier whose probe can therefore never land "exact".
        return []

    def _near_miss_candidates(self, workspace: str, token: str, limit: int) -> list[Symbol]:
        """The near-miss fallback stages for a token that FAILED resolution —
        qualifier widening first (dotted tokens), then bare-name prefix
        containment. Called only from `locate_l2`'s "none" branch (so every
        candidate here is inexact by construction); `nearest_symbol_names`
        reaches the same stages by delegating to that branch's result.

        Returns candidates ranked best-first, capped at `limit`, snippets
        populated. Empty when nothing resembles the token."""
        class_qualifier, leaf = _split_class_qualifier(token)
        # UPG-NEARMISS-WIDEN-EXACT-GUARD: only a genuine miss reaches this
        # method (the caller is the "none" branch), so widening out to OTHER
        # same-leaf classes is safe here — an exact-strategy result would have
        # resolved `token` cleanly and never been passed down.
        if class_qualifier:
            pool = self._exact_definitions(
                workspace, leaf, limit=SYMBOL_GRAPH_QUALIFIED_NEARMISS_CANDIDATE_CAP,
            )
            if not pool:
                # UPG-QUALIFIED-TYPO-NEARMISS leaf-only retry. NOTE: this
                # nested locate_l2 call must keep the default
                # `with_near_miss=False` — passing True would let the nested
                # "none" branch call back into _near_miss_candidates.
                leaf_result = self.locate_l2(
                    workspace, leaf, limit=SYMBOL_GRAPH_QUALIFIED_NEARMISS_CANDIDATE_CAP,
                )
                if leaf_result.resolution_strategy != "none":
                    pool = leaf_result.symbols
            if pool:
                # UPG-QUALIFIER-SIM-FILE-CACHE: resolve every candidate's
                # enclosing container in one batch pass (one file read per
                # unique file_path across the whole pool) rather than letting
                # `_qualifier_similarity_key` re-read the same file once per
                # candidate defined in it.
                enclosing_names = _enclosing_container_batch(pool)
                order = sorted(
                    range(len(pool)),
                    key=lambda i: self._qualifier_similarity_key(
                        class_qualifier, pool[i], enclosing_names[i],
                    ),
                )[:limit]
                top = [pool[i] for i in order]
                for i, s in zip(order, top):
                    s.snippet = self.get_snippet(s.file_path, s.start_line, s.end_line)
                    # Symbols sourced from the leaf-only `locate_l2` retry may
                    # already carry a (wrongly-anchored) qualified name from
                    # its own auto-qualification — only qualify a still-bare
                    # name here (mirrors the `_with_snippets` convention).
                    # Reuses the batch-computed enclosing name above rather
                    # than re-reading the file a second time.
                    if "." not in s.name and "::" not in s.name and enclosing_names[i]:
                        s.name = f"{enclosing_names[i]}.{s.name}"
                return top

        min_len = SEARCH_IDENTIFIER_HINT_NEARMISS_MIN_PREFIX_LEN
        if len(token) < min_len:
            return []
        token_lower = token.lower()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE workspace = ? AND LENGTH(name) >= ? "
                "AND LENGTH(name) < ? ORDER BY LENGTH(name) DESC",
                (workspace, min_len, len(token)),
            ).fetchall()
        seen_names: set[str] = set()
        candidates = []
        for r in rows:
            if r["name"] in seen_names or not token_lower.startswith(r["name"].lower()):
                continue
            seen_names.add(r["name"])
            candidates.append(r)
        if not candidates:
            return []
        syms = [self._row_to_symbol(r) for r in candidates[:limit]]
        for s in syms:
            s.snippet = self.get_snippet(s.file_path, s.start_line, s.end_line)
        return syms

    # ------------------------------------------------------------------
    # Query: trace (call graph)
    # ------------------------------------------------------------------

    # UPG-4.2: pull a wide candidate set so dedup + relevance ranking happens
    # over ALL edges, not a pre-truncated alphabetical slice (the old
    # `ORDER BY name LIMIT 20` dropped important callees by name, not relevance).
    _EDGE_FETCH_CAP = 1000

    def _edges(
        self,
        workspace: str,
        column: str,
        name: str,
        group: Literal["from_symbol", "to_symbol"],
        limit: int,
        rank_repo_defined: bool,
        include_builtins: bool = True,
        exclude_uses: bool = False,
    ) -> tuple[list[CallEdge], int, int, bool]:
        """Fetch edges by exact `column` match; fall back to partial (LIKE) only
        when no exact-named edge exists. Exact-first kills the substring
        conflation that merged unrelated symbols — `trace compare` no longer
        pulls in `compare_stacks` / `_Py_atomic_compare_exchange_*` (UPG-4.1).
        Results are deduped and ranked by relevance, then truncated (UPG-4.2).
        Returns `(edges, hidden_builtins, truncated, raw_capped)` — the count
        of builtin/stdlib callees suppressed before truncation when
        `include_builtins` is False (UPG-4.3), the count of aggregated
        entries dropped by the `limit` cut itself
        (UPG-TRACE-CALLER-LIMIT-TRANSPARENCY), and whether the SQL fetch
        itself hit `_EDGE_FETCH_CAP` so the per-direction `truncated` count
        is a LOWER BOUND and the formatter must label it as such
        (UPG-TRACE-EDGE-FETCH-CAP-UNCOUNTED). Without this flag, a heavily
        called name (more than 1000 raw edge rows) would render a "(+N
        more ...)" line that looks exact while undercounting the real
        remainder by an unbounded amount.
        `exclude_uses` drops type-usage edges (UPG-4.4) — set on the callees
        direction so "Calls:" stays function calls, not the types a function
        mentions; left off for callers so `trace <Type>` finds its usage sites.
        `column` is a fixed internal literal, never user input."""
        uses_clause = " AND edge_type != 'uses'" if exclude_uses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM edges WHERE workspace = ? AND {column} = ?{uses_clause} LIMIT ?",
                (workspace, name, self._EDGE_FETCH_CAP),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    f"SELECT * FROM edges WHERE workspace = ? AND {column} LIKE ?{uses_clause} LIMIT ?",
                    (workspace, f"%{name}%", self._EDGE_FETCH_CAP),
                ).fetchall()
        edges = [self._row_to_edge(r) for r in rows]
        shown, hidden, truncated = self._aggregate_edges(
            workspace, edges, group, limit, rank_repo_defined, include_builtins,
        )
        # UPG-TRACE-EDGE-FETCH-CAP-UNCOUNTED: a SQL fetch that returned
        # exactly `_EDGE_FETCH_CAP` rows MAY have left more on the table.
        # We don't pay a second query to know for sure — the formatter's
        # reworded note ("at least N more ...") is true under either case
        # and strictly more honest than the prior exact-looking version.
        raw_capped = len(rows) >= self._EDGE_FETCH_CAP
        return shown, hidden, truncated, raw_capped

    @staticmethod
    def _is_builtin_call(name: str, from_file: str, repo: set[str]) -> bool:
        """A callee is builtin noise only if it's a known language builtin AND not
        defined as a symbol in this repo (so a repo's own `len`/`map` stays). The
        language is inferred from the calling file's extension (UPG-4.3)."""
        if name in repo:
            return False
        from agent.indexer import LANG_BY_EXT
        lang = LANG_BY_EXT.get(Path(from_file).suffix.lower(), "")
        return name in _BUILTINS.get(lang, frozenset())

    def _aggregate_edges(
        self,
        workspace: str,
        edges: list[CallEdge],
        group: Literal["from_symbol", "to_symbol"],
        limit: int,
        rank_repo_defined: bool,
        include_builtins: bool = True,
    ) -> tuple[list[CallEdge], int, int]:
        """Collapse edges that share the same caller/callee name into one entry
        carrying a `call_count` of distinct call sites, then rank by relevance
        — repo-defined first (callees only), then call frequency, then name —
        and truncate to `limit`. Replaces the alphabetical-then-truncate path so
        important, repeatedly-called targets survive the cut (UPG-4.2).
        Returns `(edges, hidden_builtins, truncated)`.

        When `not include_builtins` (callee path only), language-builtin/stdlib
        callees are dropped *before* truncation so they can't push repo-internal
        calls out of the window; returns the count hidden (UPG-4.3). Callers are
        never filtered — a caller is by definition a repo-defined function.

        UPG-TRACE-CALLER-LIMIT-TRANSPARENCY: also returns how many aggregated,
        post-suppression entries were dropped by the `limit` cut — the number
        of real callers/callees the rendered list is silently missing. THIS is
        the authoritative layer for that count: it is the only place the
        user-facing `limit` is applied (the SQL fetch caps upstream bound raw
        rows, not aggregated entries, and every later layer sees only the
        already-truncated list), so a count computed anywhere downstream would
        itself be wrong.

        UPG-CALLER-TIEBREAK-ALPHA: on the callers path (`rank_repo_defined=
        False`) call frequency was the only relevance signal above — a tied
        call_count fell all the way through to plain alphabetical name order.
        Ties are now broken by the caller's own repo-defined ancestry
        (repo-defined before external/unresolved — `ingest_trace_data`
        documents that a caller name need not resolve to a known symbol),
        then by in-degree — how many distinct places elsewhere in the graph
        call that candidate — as a bounded, query-time stand-in for
        centrality; no new stored/precomputed centrality pipeline. Both
        lookups are scoped to only the names that actually collide on
        call_count, so the common untied case pays no extra query cost.

        UPG-TRACE-CALLER-AGGREGATION: the collapse key on the callers path
        (`group="from_symbol"`) used to be the bare name alone — two
        unrelated functions sharing a leaf name in different files (e.g.
        `__call__` defined on two different classes) merged every call site
        under ONE elected representative edge, so a real caller in the
        second file never appeared anywhere in the rendered trace even
        though its call was counted in the `×N` suffix. Callers now
        collapse by (name, from_file): call sites in the SAME file still
        fold into one entry with a `call_count` (a real ambiguity a
        receiver-blind static graph can't resolve further — which exact
        call in that file bound the reference), but two callers sharing a
        name in DIFFERENT files are always kept as distinct entries, so
        neither can hide the other. `from_file` is the final, deterministic
        tiebreak when two such entries also tie on every other signal.
        Callees (`group="to_symbol"`) keep the name-only collapse — one
        aggregation call here only ever covers callees of a SINGLE
        definition's call sites, so `to_symbol`'s originating file never
        varies within it; a callEE name itself resolving to multiple
        DEFINITIONS is a different ambiguity, already handled separately by
        `trace()`'s `by_definition` split (UPG-4.1)."""
        groups: dict[tuple | str, dict] = {}
        for e in edges:
            name = e.from_symbol if group == "from_symbol" else e.to_symbol
            ident = (name, e.from_file) if group == "from_symbol" else name
            site = (e.from_file, e.from_line, e.to_symbol)
            g = groups.get(ident)
            if g is None:
                groups[ident] = {"edge": e, "sites": {site}, "name": name}
            else:
                g["sites"].add(site)
        # Repo-defined ranking only matters for callees (the *from_symbol* of a
        # caller is by definition a function in this repo). Skipping the lookup
        # for callers also avoids a needless symbols-table scan. Deduped by
        # name — two caller groups sharing a name (different files) must not
        # bind the same symbols-table row twice.
        names = list({g["name"] for g in groups.values()})
        repo = self._known_symbol_names(workspace, names) if rank_repo_defined else set(names)
        suppress = rank_repo_defined and not include_builtins

        tie_repo: set[str] = set()
        tie_in_degree: dict[str, int] = {}
        if not rank_repo_defined:
            tied = self._tied_call_count_names(groups)
            if tied:
                tie_repo = self._known_symbol_names(workspace, list(tied))
                tie_in_degree = self._caller_in_degree(workspace, list(tied))

        ranked: list[tuple] = []
        hidden = 0
        for g in groups.values():
            e = g["edge"]
            name = g["name"]
            if suppress and self._is_builtin_call(name, e.from_file, repo):
                hidden += 1
                continue
            e.call_count = len(g["sites"])
            if rank_repo_defined:
                ranked.append((0 if name in repo else 1, -e.call_count, name, e))
            else:
                ranked.append((
                    -e.call_count,
                    0 if name in tie_repo else 1,
                    -tie_in_degree.get(name, 0),
                    name,
                    e.from_file,
                    e,
                ))
        ranked.sort(key=lambda t: t[:-1])
        shown = [t[-1] for t in ranked[:limit]]
        # UPG-TRACE-CALLER-LIMIT-TRANSPARENCY: entries beyond the cut are real
        # callers/callees the rendered list will not show — counted HERE, at
        # the only layer that applies the user-facing limit.
        truncated = max(0, len(ranked) - limit)
        return shown, hidden, truncated

    @staticmethod
    def _tied_call_count_names(groups: dict) -> set[str]:
        """Names in *groups* whose call_count collides with at least one
        other name's — the only candidates a tie-break signal can actually
        change the order of (UPG-CALLER-TIEBREAK-ALPHA); an untied name is
        already fully decided by call_count and never consults these maps.
        Reads each group's own `name` field rather than its dict key
        (UPG-TRACE-CALLER-AGGREGATION: the key is `(name, from_file)` on the
        callers path, no longer the bare name alone) — two groups for the
        SAME name in different files correctly count as a tie needing a
        secondary signal, same as two groups for different names."""
        by_count: dict[int, list[str]] = {}
        for g in groups.values():
            by_count.setdefault(len(g["sites"]), []).append(g["name"])
        return {name for names in by_count.values() if len(names) > 1 for name in names}

    def _caller_in_degree(self, workspace: str, names: list[str]) -> dict[str, int]:
        """How many distinct calling functions target each of *names*
        elsewhere in the graph — a bounded, query-time proxy for a
        candidate caller's own importance/centrality (UPG-CALLER-TIEBREAK-
        ALPHA). Deliberately not a new stored centrality pipeline: one
        grouped COUNT over the same `edges` table `_known_symbol_names`
        already queries (using the same `idx_edge_to` index), scoped to the
        (typically small) set of names actually tied on call_count.

        UPG-CALLER-INDEGREE-ANALYZE: this `idx_edge_to` plan only holds when
        SQLite has table/index statistics to reason with — `build_for_
        workspace` runs a bounded `ANALYZE edges` once per full build so a fresh or
        rebuilt graph always has them; without it SQLite's planner falls
        back to a full covering-index workspace scan here (measured ~15x
        slower on a synthetic 300k-edge graph)."""
        if not names:
            return {}
        counts: dict[str, int] = {}
        with self._conn() as conn:
            for i in range(0, len(names), 500):  # stay under SQLite's bound-var limit
                chunk = names[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT to_symbol, COUNT(DISTINCT from_symbol) AS c FROM edges "
                    f"WHERE workspace = ? AND to_symbol IN ({placeholders}) "
                    f"GROUP BY to_symbol",
                    (workspace, *chunk),
                ).fetchall()
                for r in rows:
                    counts[r["to_symbol"]] = r["c"]
        return counts

    def _known_symbol_names(self, workspace: str, names: list[str]) -> set[str]:
        """Subset of `names` that are defined as symbols in this workspace —
        used to rank repo-internal calls ahead of builtins/externals (UPG-4.2)."""
        if not names:
            return set()
        found: set[str] = set()
        with self._conn() as conn:
            for i in range(0, len(names), 500):  # stay under SQLite's bound-var limit
                chunk = names[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT DISTINCT name FROM symbols WHERE workspace = ? "
                    f"AND name IN ({placeholders})",
                    (workspace, *chunk),
                ).fetchall()
                found.update(r["name"] for r in rows)
        return found

    def callers(self, workspace: str, symbol_name: str, limit: int = 20) -> list[CallEdge]:
        """Who calls this symbol? Exact name match preferred (partial fallback).
        Deduped per (caller name, file), ranked by call frequency (UPG-4.2)."""
        edges, _, _, _ = self._edges(workspace, "to_symbol", symbol_name, "from_symbol", limit, rank_repo_defined=False)
        return edges

    def callees(
        self, workspace: str, symbol_name: str, limit: int = 20, include_builtins: bool = True
    ) -> list[CallEdge]:
        """What does this symbol call? Exact name match preferred (partial fallback).
        Deduped by callee, repo-internal calls ranked ahead of builtins (UPG-4.2);
        builtin/stdlib callees suppressed unless `include_builtins` (UPG-4.3)."""
        edges, _, _, _ = self._edges(
            workspace, "from_symbol", symbol_name, "to_symbol", limit,
            rank_repo_defined=True, include_builtins=include_builtins,
            exclude_uses=True,
        )
        return edges

    # ------------------------------------------------------------------
    # UPG-BASE-METHOD-OVERRIDE-FLOOD: read API for same-file override edges
    # written by `_extract_overrides` at index time. The data these return
    # is the input a later ranking change would consume to lift a base
    # method above its overrides in a same-leaf top-k tie (the F1/F23/
    # F56 family). This lane ships the read API; the ranking change is
    # out of scope and must be evidence-gated on real measurement.
    # ------------------------------------------------------------------

    def overrides_of(
        self,
        workspace: str,
        from_file: str,
        from_symbol: str,
        from_line: int,
    ) -> list[CallEdge]:
        """Same-file base methods that *from_symbol* at *from_file:from_line*
        overrides (UPG-BASE-METHOD-OVERRIDE-FLOOD).

        The source is identified by (file, symbol name, line) — the same
        triple the existing `edges` table uses for source identity. Returns
        one CallEdge per base the method overrides; a method that overrides
        multiple bases (e.g. a TS `class X extends A, B, C` where A, B, and
        C are in the same file) returns one edge per base. A method that
        extends a base defined in a different file is not present in the
        result — see LANE-REPORT for the same-file scope rationale.

        The returned `to_symbol` is the qualified `BaseClass.method` form
        so the caller can `locate()` it without ambiguity. The base
        file is implicit (same as `from_file` by construction).
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT from_file, from_symbol, from_line, to_symbol, edge_type "
                "FROM edges WHERE workspace = ? AND edge_type = 'overrides' "
                "AND from_file = ? AND from_symbol = ? AND from_line = ?",
                (workspace, from_file, from_symbol, from_line),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def overridden_by(
        self,
        workspace: str,
        to_file: str,
        to_symbol: str,
    ) -> list[CallEdge]:
        """Same-file subclass methods that override *to_symbol* in *to_file*
        (UPG-BASE-METHOD-OVERRIDE-FLOOD). The inverse of `overrides_of`.

        `to_symbol` is the qualified `BaseClass.method` form (as written
        by `_extract_overrides`). Returns one CallEdge per subclass method
        that overrides this base; the source `(from_file, from_symbol,
        from_line)` triple on the returned edge identifies each override.
        `from_file` equals `to_file` by the same-file scope of this lane.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT from_file, from_symbol, from_line, to_symbol, edge_type "
                "FROM edges WHERE workspace = ? AND edge_type = 'overrides' "
                "AND from_file = ? AND to_symbol = ?",
                (workspace, to_file, to_symbol),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def _exact_definitions(self, workspace: str, name: str, limit: int = 20) -> list[Symbol]:
        """Definition sites whose name matches `name` exactly. Each (file_path,
        name) is a distinct node — this is the fully-qualified identity that
        keeps same-named symbols in different modules from merging (UPG-4.1).

        UPG-TRACE-GRAPH-INCOMPLETE (F60): ordered by `_CANONICAL_FETCH_ORDER`
        (non-test file first, larger line-span first) rather than plain
        `file_path` — a common leaf name (e.g. `delete`) can have more
        definitions than `limit` across a large codebase, and a bare
        alphabetical-by-path order truncates on file-path spelling alone.
        one module's `Repository.delete` sorted after another module's
        `Model.delete` purely because "r" > "m",
        silently dropping it from every trace path that reaches this method —
        not a corpus-specific fix; any workspace with >`limit` definitions of
        one leaf name hits the same alphabetical-accident truncation."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE workspace = ? AND name = ? "
                + _CANONICAL_FETCH_ORDER + " LIMIT ?",
                (workspace, name, limit),
            ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    @staticmethod
    def _qualifier_similarity_key(
        class_qualifier: str, candidate: Symbol, enclosing: str,
    ) -> tuple[float, float]:
        """Sort key for UPG-LOCATE-FALLBACK-NO-SIMILARITY (ascending order =
        best match first — negate the ratios so `sorted()` puts the closest
        candidate first). Ranks a same-leaf definition candidate by how
        closely its OWN enclosing type resembles the class qualifier that
        failed to resolve — a fabricated or misremembered qualifier is
        usually a near-miss of a real class/module name elsewhere, not
        random. `difflib.SequenceMatcher` is deterministic stdlib string
        similarity, never a semantic guess.

        Enclosing-type similarity is primary; file-path similarity is a
        secondary signal only consulted on a tie (e.g. a candidate with no
        enclosing type at all — a module-level definition — falls back to
        comparing the qualifier against its path). The caller's `sorted()`
        is stable, so a full tie on both ratios preserves the candidates'
        original (canonical-fetch-order) relative order.

        UPG-QUALIFIER-SIM-FILE-CACHE: `enclosing` is precomputed by the
        caller via `_enclosing_container_batch` (one file read per unique
        file_path across the whole ranking pool) rather than derived here
        via `_enclosing_container_from_file` — a per-candidate call would
        re-read (and, on the impl-container fallback, re-parse) the same
        file once per candidate defined in it, up to the qualified near-miss
        candidate pool cap file reads on this fallback path."""
        type_ratio = (
            SequenceMatcher(None, class_qualifier.lower(), enclosing.lower()).ratio()
            if enclosing else 0.0
        )
        path_ratio = SequenceMatcher(
            None, class_qualifier.lower(), candidate.file_path.lower()
        ).ratio()
        return (-type_ratio, -path_ratio)

    @staticmethod
    def _module_label(file_path: str, workspace: str) -> str:
        """Repo-relative path used to qualify a definition in trace output."""
        try:
            return str(Path(file_path).relative_to(workspace))
        except ValueError:
            return Path(file_path).name

    def trace(
        self,
        workspace: str,
        symbol_name: str,
        direction: Literal["callers", "callees", "both"] = "both",
        limit: int = 20,
        include_builtins: bool = False,
    ) -> dict:
        """Combined callers + callees lookup.

        UPG-4.1: when `symbol_name` has more than one definition across modules,
        the callees are scoped per definition (by `from_file`) so they are shown
        separately instead of merged into one node. Callees are exactly
        attributable (an edge carries the calling definition's file); callers are
        not (a call site doesn't record which definition it bound), so callers
        stay a flat list with an ambiguity note in the formatter.

        UPG-4.3: builtin/stdlib callees are hidden by default; the count hidden is
        recorded under `hidden_builtins` (flat) / per `by_definition` entry so the
        formatter can offer `include_builtins`.

        F33 (UPG-17.1): `symbol_name` may arrive in the qualified "Class.method"
        form — vectr_search displays symbols that way, and an LLM naturally
        copies the displayed name into a follow-up trace call. The edge/symbol
        tables only ever store the bare leaf (`_split_class_qualifier` is the
        shared split also used by `locate_l2`), so a literal `name = "Class.method"`
        match against either table is always empty. Resolution here: look up
        callers/callees/definitions by the bare leaf, then — when a class
        qualifier was given — filter `definitions` down to the site(s) whose
        enclosing class matches and scope `by_definition`'s callees to exactly
        those sites (the same per-definition attribution UPG-4.1 already does
        for an unqualified ambiguous name), so a qualified query gets a
        precise answer instead of the leaf's merged-across-classes one.

        UPG-TRACE-CALLERS-QUALIFIER-VALIDATION: a qualifier that names no real
        class enclosing the leaf (a fabricated or misremembered "Class.method")
        must not be silently answered with the bare leaf's flat callers/callees
        as though they belonged to that class — a caller LLM has no way to
        tell a validated answer from an unvalidated leaf fallback otherwise.
        `resolved_qualifier` below is the SAME enclosing-class check already
        used to scope `definitions`/`by_definition`; when it fails, that
        failure is recorded (`qualifier_unresolved`) and near-miss qualified
        names are looked up (reusing `nearest_symbol_names`, the same
        machinery `locate` already uses) so the formatter can lead with an
        honest miss instead of a confident-looking wrong answer.
        """
        class_qualifier, leaf = _split_class_qualifier(symbol_name)
        lookup_name = leaf if class_qualifier else symbol_name

        result: dict = {}
        # UPG-TRACE-DIRECTION-UNQUERIED-RENDER: a direction the caller did NOT
        # ask for is recorded as `None`, not `[]`, so the formatter can
        # distinguish "we looked and found nothing" from "we never looked".
        # `direction="both"` legitimately renders both blocks when both are
        # genuinely empty — the `None`/`[]` split is what keeps that honest
        # without conflating it with a single-direction trace.
        if direction in ("callers", "both"):
            # Callers can't be attributed to one class-qualified definition —
            # a call site never records the receiver's type — so a resolved
            # qualified query still returns the leaf's full (ambiguous) caller
            # list rather than a silent empty result. When the qualifier does
            # NOT resolve at all (below), the formatter is told so it can
            # label this same list honestly instead of presenting it as the
            # named class's callers.
            #
            # UPG-TRACE-CALLER-LIMIT-TRANSPARENCY: called via `_edges` rather
            # than `self.callers()` only to capture how many aggregated caller
            # entries the `limit` cut dropped — the formatter reports it.
            # UPG-TRACE-EDGE-FETCH-CAP-UNCOUNTED: also captures whether the
            # SQL fetch itself hit the upstream cap so the formatter can
            # label the rendered "more not shown" count as a lower bound.
            callers, _, callers_truncated, callers_raw_capped = self._edges(
                workspace, "to_symbol", lookup_name, "from_symbol", limit,
                rank_repo_defined=False,
            )
            result["callers"] = callers
            result["callers_truncated"] = callers_truncated
            result["callers_raw_capped"] = callers_raw_capped
        else:
            result["callers"] = None  # UPG-TRACE-DIRECTION-UNQUERIED-RENDER
        if direction in ("callees", "both"):
            callees, hidden, callees_truncated, callees_raw_capped = self._edges(
                workspace, "from_symbol", lookup_name, "to_symbol", limit,
                rank_repo_defined=True, include_builtins=include_builtins,
                exclude_uses=True,
            )
            result["callees"] = callees
            result["hidden_builtins"] = hidden
            result["callees_truncated"] = callees_truncated
            result["callees_raw_capped"] = callees_raw_capped
        else:
            result["callees"] = None  # UPG-TRACE-DIRECTION-UNQUERIED-RENDER

        defs = self._exact_definitions(workspace, lookup_name, limit=limit)
        # UPG-TRACE-CALLER-LIMIT-TRANSPARENCY: the definitions fetch is capped
        # at `limit` too, and each definition renders as its own by-definition
        # section — when the cap bit, whole definitions (with their calls) are
        # silently missing. One conditional COUNT query resolves whether the
        # fetch actually cut anything (len == limit could also mean exactly
        # limit exist); zero cost on every non-boundary trace.
        definitions_truncated = 0
        if len(defs) == limit:
            with self._conn() as conn:
                total_defs = conn.execute(
                    "SELECT COUNT(*) AS c FROM symbols WHERE workspace = ? AND name = ?",
                    (workspace, lookup_name),
                ).fetchone()["c"]
            definitions_truncated = max(0, total_defs - len(defs))
        result["definitions_truncated"] = definitions_truncated
        resolved_qualifier = False
        if class_qualifier:
            # UPG-RUST-IMPL-QUALIFIED-LOCATE: owner may be a class OR an
            # impl-style block (Rust) — `_enclosing_container_from_file`
            # covers both, unlike the class-keyword-only text scan.
            filtered = [
                d for d in defs
                if _enclosing_container_from_file(d.file_path, d.start_line) == class_qualifier
            ]
            if filtered:
                defs = filtered
                resolved_qualifier = True
                for d in defs:
                    d.name = f"{class_qualifier}.{d.name}"
        result["definitions"] = defs
        if resolved_qualifier:
            result["qualified_class"] = class_qualifier
        elif class_qualifier:
            # The qualifier was given but matched no real definition's
            # enclosing class — every leaf-scoped answer below (callers, and
            # callees/definitions when the by-definition breakdown does not
            # itself fire) is unvalidated against it.
            result["qualifier_unresolved"] = True
            result["qualifier_near_miss"] = self.nearest_symbol_names(
                workspace, symbol_name, limit=SYMBOL_GRAPH_TRACE_QUALIFIER_NEARMISS_MAX
            )

        if direction in ("callees", "both") and (len(defs) > 1 or resolved_qualifier):
            by_def = []
            with self._conn() as conn:
                for d in defs:
                    rows = conn.execute(
                        "SELECT * FROM edges WHERE workspace = ? AND from_symbol = ? "
                        "AND from_file = ? AND edge_type != 'uses' LIMIT ?",
                        (workspace, lookup_name, d.file_path, self._EDGE_FETCH_CAP),
                    ).fetchall()
                    edges = [self._row_to_edge(r) for r in rows]
                    # dedup + relevance-rank + builtin-suppress this def's callees
                    cs, hidden, def_truncated = self._aggregate_edges(
                        workspace, edges, "to_symbol", limit,
                        rank_repo_defined=True, include_builtins=include_builtins,
                    )
                    by_def.append({
                        "definition": d,
                        "module": self._module_label(d.file_path, workspace),
                        "callees": cs,
                        "hidden_builtins": hidden,
                        "truncated_callees": def_truncated,  # UPG-TRACE-CALLER-LIMIT-TRANSPARENCY
                        # UPG-TRACE-EDGE-FETCH-CAP-UNCOUNTED: same per-entry
                        # raw-cap signal as the flat direction — when this
                        # definition alone has more raw edges than the SQL
                        # cap, the per-entry "(+N more ...)" line is a
                        # lower bound, not a count.
                        "truncated_callees_raw_capped": len(rows) >= self._EDGE_FETCH_CAP,
                    })
            result["by_definition"] = by_def
        return result

    def get_snippet(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Read up to SNIPPET_LINES from a file starting at start_line (1-indexed).
        Returns the raw code so the AI editor can read and understand it directly.
        """
        try:
            lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
            s = max(0, start_line - 1)
            e = min(len(lines), s + SNIPPET_LINES)
            snippet = "\n".join(lines[s:e])
            logger.debug("get_snippet: %s:%d-%d (%d lines)", file_path, start_line, end_line, e - s)
            return snippet
        except OSError as exc:
            logger.warning("get_snippet: could not read %s — %s", file_path, exc)
            return ""

    def symbols_touching_file(self, workspace: str, file_path: str) -> frozenset[str]:
        """Every symbol name DEFINED IN or REFERENCED BY `file_path` — the
        one-graph-lookup-per-target-file resolution the S (symbol) trigger
        primitive needs (TRIGGER-ENGINE wave 2b, bm2-design-skeleton.md §2).
        Called ONCE per lifecycle moment by the caller (`WorkingContextStore
        .fire()`); every individual note's symbol trigger is then a plain
        set-membership check against this result — never a separate graph
        query per note."""
        with self._conn() as conn:
            defined = conn.execute(
                "SELECT DISTINCT name FROM symbols WHERE workspace = ? AND file_path = ?",
                (workspace, file_path),
            ).fetchall()
            referenced = conn.execute(
                "SELECT DISTINCT to_symbol FROM edges WHERE workspace = ? AND from_file = ?",
                (workspace, file_path),
            ).fetchall()
        return frozenset(r[0] for r in defined) | frozenset(r[0] for r in referenced)

    def signature_hash(self, workspace: str, name: str) -> str | None:
        """sha256[:16] of `name`'s canonical definition snippet — the
        write-time anchor a symbol-triggered note records (TRIGGER-ENGINE
        wave 2b §5, bm2-design-skeleton.md), mirroring `_hash_path_content()`
        's content-hash path anchor in WorkingContextStore. Uses the SAME
        "pick ONE definition" `_exact_definitions()` ordering `locate`/
        `trace` already use, so the hash is stable across repeated calls
        against an unchanged codebase. Returns None when `name` has no
        resolvable definition (or its snippet reads back empty) — a note
        recording a hash of None never raises a staleness caveat later
        (nothing to compare against); only a genuine content change that WAS
        captured at write time does."""
        definitions = self._exact_definitions(workspace, name, limit=1)
        if not definitions:
            return None
        d = definitions[0]
        snippet = self.get_snippet(d.file_path, d.start_line, d.end_line)
        if not snippet:
            return None
        return hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Formatting for LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_description(query: str) -> bool:
        """A no-match query that reads like a natural-language description rather
        than a symbol name — the LLM likely misrouted a `vectr_search` query to
        `locate`. Whitespace between tokens is the tell: `is_prime` won't fire,
        but "function that checks whether a number is prime" will (UPG-4.6)."""
        return len(query.split()) > 1

    def _no_match_text(self, query: str) -> str:
        """Empty-locate message that ALWAYS hands the LLM a path forward — never a
        dead end (UPG-10.3, extends UPG-4.6). A silent no-match trains the model
        to abandon `locate` and fall back to grep; a redirect keeps it on a vectr
        tool. Description-shaped misses point at the misroute (UPG-4.6); a plain
        single-token miss points at content search, since the name may be a kind
        the symbol graph doesn't make locatable or simply isn't present."""
        base = f"No symbol matching '{query}' found in the indexed codebase."
        if self._looks_like_description(query):
            return base + (" This looks like a description, not a symbol name — "
                           "try vectr_search for concept/semantic lookup.")
        return base + (f' Try vectr_search("{query}") to find it by content — it '
                       "may be defined under a different name or not indexed as a "
                       "locatable symbol.")

    def format_locate_for_llm(self, symbols: list[Symbol], name: str) -> str:
        if not symbols:
            return self._no_match_text(name)
        lines = [f"Symbol locations for '{name}' ({len(symbols)} match{'es' if len(symbols) != 1 else ''}):\n"]
        for s in symbols:
            lines.append(f"  [{s.kind}] {s.name}  {s.file_path}:{s.start_line}")
            if s.snippet:
                for ln in s.snippet.splitlines()[:SNIPPET_LINES]:
                    lines.append(f"    {ln}")
                lines.append("")
        return "\n".join(lines)

    def format_locate_l2_for_llm(self, result: LocateResult) -> str:
        if not result.symbols:
            # UPG-LOCATE-NEARMISS-WIRE-PRIMARY-TOOL: a total miss that carries
            # near-miss candidates renders them explicitly labeled inexact,
            # following the labeling convention the other two surfaces already
            # established — trace()'s qualifier-near-miss banner ("nearest ...
            # by name resolution (inexact — verify before use)") and the
            # search-hint section ("No exact match for '...'; nearest symbol
            # names") — with the same listing shape as the fuzzy block below.
            # Never a silent substitution: the miss stays a miss.
            if result.near_miss:
                names = ", ".join(s.name for s in result.near_miss)
                n = len(result.near_miss)
                lines = [
                    f"No exact match for '{result.query}'. "
                    f"Nearest symbol name{'s' if n != 1 else ''} by name resolution "
                    f"(inexact — verify before use): {names}\n"
                ]
                for s in result.near_miss:
                    lines.append(f"  [{s.kind}] {s.name}  {s.file_path}:{s.start_line}")
                    if s.snippet:
                        for ln in s.snippet.splitlines()[:SNIPPET_LINES]:
                            lines.append(f"    {ln}")
                        lines.append("")
                return "\n".join(lines)
            return self._no_match_text(result.query)
        _labels = {
            "exact":        "exact name match",
            "suffix":       "suffix match (qualifier stripped)",
            "same_module":  "same-module resolution",
            "substring":    "partial-name match (canonical defs first)",
            "import_chain": "import-chain resolution",
            "fuzzy":        "fuzzy name match (edit-distance)",
        }
        n = len(result.symbols)
        # UPG-C-STRUCT-TYPEDEF-LOCATE (fuzzy-fallback honesty, general): an
        # edit-distance match is a GUESS, not a match — a confident-looking
        # wrong symbol (`locate("PyDictObject")` → `PyODictObject`) is worse than
        # not-found because the caller LLM acts on it. Lead with an explicit
        # no-exact-match caveat so the near-miss is never read as a real hit.
        if result.resolution_strategy == "fuzzy":
            names = ", ".join(s.name for s in result.symbols)
            lines = [
                f"No exact match for '{result.query}'. "
                f"Nearest symbol name{'s' if n != 1 else ''} by edit-distance "
                f"(may be unrelated — verify before use): {names}\n"
            ]
            for s in result.symbols:
                lines.append(f"  [{s.kind}] {s.name}  {s.file_path}:{s.start_line}")
                if s.snippet:
                    for ln in s.snippet.splitlines()[:SNIPPET_LINES]:
                        lines.append(f"    {ln}")
                    lines.append("")
            return "\n".join(lines)
        label = _labels.get(result.resolution_strategy, result.resolution_strategy)
        lines = [
            f"Symbol locations for '{result.query}' "
            f"({n} match{'es' if n != 1 else ''} via {label}):\n"
        ]
        for s in result.symbols:
            lines.append(f"  [{s.kind}] {s.name}  {s.file_path}:{s.start_line}")
            if s.snippet:
                for ln in s.snippet.splitlines()[:SNIPPET_LINES]:
                    lines.append(f"    {ln}")
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _count_suffix(edge: CallEdge) -> str:
        """' ×N' when an aggregated edge stands for multiple call sites (UPG-4.2)."""
        return f"  ×{edge.call_count}" if edge.call_count > 1 else ""

    @staticmethod
    def _dynamic_marker(edge: CallEdge) -> str:
        """' (dynamic)' for an edge ingested via vectr_ingest_traces rather than
        found by static analysis (UPG-7.3) — tells the LLM the edge came from
        runtime observation (e.g. __getattr__, decorators, dependency injection)
        and may not be visible by reading the source around the call site."""
        return "  (dynamic)" if getattr(edge, "edge_type", "calls") == "dynamic" else ""

    @staticmethod
    def _caller_verb(callers: list) -> str:
        """'Used by' when every reference is a type-usage edge (UPG-4.4) — e.g.
        tracing a Rust struct that's only passed/returned, never free-called;
        'Called/used by' when mixed; 'Called by' otherwise."""
        kinds = {getattr(e, "edge_type", "calls") for e in callers}
        if kinds == {"uses"}:
            return "Used by"
        if "uses" in kinds:
            return "Called/used by"
        return "Called by"

    @staticmethod
    def _hidden_builtins_note(n: int) -> str:
        """Footer telling the LLM repo-internal calls are shown and how to see the
        rest — the suppressed builtin/stdlib calls (UPG-4.3)."""
        if n <= 0:
            return ""
        return (f"    (+{n} builtin/stdlib call{'s' if n != 1 else ''} hidden — "
                f"pass include_builtins=true to show)")

    @staticmethod
    def _truncation_note(n: int, what: str, raw_capped: bool = False) -> str:
        """Footer telling the LLM how many entries a `limit` cut removed from
        the list just rendered — without it, a caller/callee list reads as
        complete when it is only a prefix of the real one, and the caller has
        no signal to act on (UPG-TRACE-CALLER-LIMIT-TRANSPARENCY). Same shape
        as the hidden-builtins footer above.

        UPG-TRACE-EDGE-FETCH-CAP-UNCOUNTED: when `raw_capped` is True the
        SQL fetch upstream of aggregation also hit `_EDGE_FETCH_CAP`, so
        this `n` is a LOWER BOUND on the real remainder (more raw rows
        may have been dropped before aggregation even ran). The note
        rewords to "at least N more" and stops suggesting that raising
        the user-facing `limit` is a sufficient fix — the cap that has
        to be lifted for an exact count is the SQL `_EDGE_FETCH_CAP`
        (static; not user-tunable today)."""
        if n <= 0:
            return ""
        if raw_capped:
            return (
                f"    (+at least {n} more {what} not shown — raw edge row cap "
                f"may hide more; count is a lower bound, not exact)"
            )
        return f"    (+{n} more {what} not shown — pass a higher limit to show)"

    @staticmethod
    def _empty_trace_hint(symbol_name: str) -> str:
        """Recovery hint for a trace with NO callers and NO callees found
        (UPG-TRACE-EMPTY-HINT, case F61) — mirrors the redirect other empty
        results already give: locate-miss suggests vectr_search (`_no_match_text`
        above), search low-confidence suggests vectr_locate (mcp_server/
        `_dispatch.py`). A bare '(none found in index)' with no follow-up trains
        the model to abandon `trace` silently; static call-graph analysis can't
        see dynamic dispatch (attribute calls on an instance, decorators,
        dependency injection), so a real call site can exist and still be
        invisible here."""
        return (
            f"\nStatic call-graph analysis found no calls for '{symbol_name}'. "
            f"If it's invoked via dynamic dispatch (e.g. an instance attribute "
            f"call, a decorator, or dependency injection) rather than a direct "
            f"name reference, try vectr_search(query=\"{symbol_name} usage\") "
            f"or vectr_search(query=\"{symbol_name} call site\") to find call "
            f"sites by content instead."
        )

    @staticmethod
    def _class_trace_note() -> str:
        """Note appended when the traced symbol is a class definition with no
        callees (UPG-TRACE-EMPTY-HINT) — a class itself is never "called"; its
        methods are. Without this, calls(0) on a class reads like a real
        no-calls-found result rather than tracing the wrong granularity
        (CursorDebugWrapper witness: class-level trace never aggregates its
        methods' calls)."""
        return (
            "Note: this is a class — tracing the class name itself does not "
            "aggregate its methods' calls. Trace a specific method name "
            "(e.g. vectr_trace(name=\"MethodName\")) to see its call details."
        )

    @staticmethod
    def _qualifier_unresolved_banner(
        symbol_name: str, class_qualifier: str, leaf_name: str, near_miss: list,
    ) -> str:
        """Leading honesty banner for UPG-TRACE-CALLERS-QUALIFIER-VALIDATION: a
        `Class.method` qualifier that matched no real definition must say so
        BEFORE any leaf-scoped fallback data, not after — otherwise a caller
        LLM reading top-to-bottom sees a confident-looking answer before it
        would ever reach a caveat trailing it. `near_miss` is additive and
        deterministic (`SymbolGraph.nearest_symbol_names`, the same machinery
        `locate` already uses for a near-miss) — never a semantic guess, and
        empty when no near-miss candidate exists."""
        lines = [
            f"⚠ '{symbol_name}' does not resolve — no definition of "
            f"'{leaf_name}' found inside a class named '{class_qualifier}'."
        ]
        if near_miss:
            names = ", ".join(s.name for s in near_miss)
            lines.append(
                f"  Nearest real symbol name(s) by name resolution "
                f"(inexact — verify before use): {names}"
            )
        lines.append("")
        return "\n".join(lines)

    def _trace_call_site_ids(
        self,
        workspace: str,
        callers: list | None,
        callee_lists: list[list],
        chunk_span_lookup: (
            "Callable[[list[tuple[str, int]]], dict[tuple[str, int], tuple[int, int]]] | None"
        ),
    ) -> tuple[dict[tuple[str, int], tuple[int, int]], dict[str, tuple[str, int]]]:
        """Batch-resolve real, stored chunk spans for every call site this
        response will render (UPG-TRACE-FETCH-IDS) — ONE call to
        `chunk_span_lookup` covering every caller's own call site plus every
        callee's own (unambiguous) definition site, never a per-line query.
        This is the response-packing lens: a caller/callee line that already
        names a location gets the `vectr_fetch`-ready id for it in the SAME
        response, deterministic and token-bounded, instead of forcing a
        follow-up `vectr_locate` round-trip per name.

        Returns `(spans, callee_loc)`:
          `spans`      — (file_path, line) -> the real (start_line, end_line)
                         of the chunk covering that line, ONLY for locations
                         the index actually has a stored chunk for. A
                         location the resolver can't match is simply absent
                         — the render helpers below treat that as "no id",
                         never a guessed one.
          `callee_loc` — to_symbol name -> (file_path, start_line) of its
                         OWN definition, resolved via the same exact-name
                         lookup used elsewhere in this class, and ONLY
                         populated when the name resolves to exactly one
                         definition (an ambiguous callee name gets no id
                         rather than an arbitrarily-chosen one).
        """
        if chunk_span_lookup is None or not workspace:
            return {}, {}
        locations: set[tuple[str, int]] = set()
        for e in callers or []:
            locations.add((e.from_file, e.from_line))
        callee_loc: dict[str, tuple[str, int]] = {}
        for cs in callee_lists:
            for e in cs:
                if e.to_symbol in callee_loc:
                    continue
                defs = self._exact_definitions(workspace, e.to_symbol, limit=2)
                if len(defs) == 1:
                    loc = (defs[0].file_path, defs[0].start_line)
                    callee_loc[e.to_symbol] = loc
                    locations.add(loc)
        if not locations:
            return {}, callee_loc
        try:
            spans = chunk_span_lookup(list(locations))
        except Exception:
            spans = {}
        return spans, callee_loc

    @staticmethod
    def _fetch_id_suffix(
        location: tuple[str, int] | None,
        spans: dict[tuple[str, int], tuple[int, int]],
        workspace: str,
    ) -> str:
        """`  [fetch: relpath:start-end]` for a location whose real, stored
        chunk span resolved, else "" — never a fabricated id
        (UPG-TRACE-FETCH-IDS): a plausible-looking id that doesn't actually
        resolve through `vectr_fetch` is worse than no id at all."""
        if location is None:
            return ""
        span = spans.get(location)
        if span is None:
            return ""
        file_path, _ = location
        rel = workspace_relpath(file_path, workspace)
        return f"  [fetch: {rel}:{span[0]}-{span[1]}]"

    def _render_caller_line(
        self, e: CallEdge, spans: dict[tuple[str, int], tuple[int, int]], workspace: str,
    ) -> str:
        """One `Called by` line — UPG-TRACE-ABS-PATH renders `e.from_file`
        workspace-relative; UPG-TRACE-FETCH-IDS appends a real fetch id when
        the call site resolves to a stored chunk."""
        rel = workspace_relpath(e.from_file, workspace)
        suffix = self._fetch_id_suffix((e.from_file, e.from_line), spans, workspace)
        return (
            f"  {e.from_symbol}  in {rel}:{e.from_line}"
            f"{self._count_suffix(e)}{self._dynamic_marker(e)}{suffix}"
        )

    def _render_callee_line(
        self,
        e: CallEdge,
        indent: str,
        callee_loc: dict[str, tuple[str, int]],
        spans: dict[tuple[str, int], tuple[int, int]],
        workspace: str,
    ) -> str:
        """One `Calls` line — UPG-TRACE-FETCH-IDS appends a real fetch id for
        the callee's own definition when it resolves unambiguously."""
        suffix = self._fetch_id_suffix(callee_loc.get(e.to_symbol), spans, workspace)
        return f"{indent}{e.to_symbol}{self._count_suffix(e)}{self._dynamic_marker(e)}{suffix}"

    def format_trace_for_llm(
        self,
        trace_result: dict,
        symbol_name: str,
        workspace: str = "",
        chunk_span_lookup: (
            "Callable[[list[tuple[str, int]]], dict[tuple[str, int], tuple[int, int]]] | None"
        ) = None,
    ) -> str:
        # UPG-TRACE-ABS-PATH: the absolute workspace root is printed ONCE
        # here (when known) — every path/id rendered below is relative to
        # it, the same convention `_format_search_results` and
        # `EvictionAdvisor` already use (UPG-RELATIVE-PATH-RENDER).
        lines = [f"Call graph trace for '{symbol_name}':"]
        if workspace:
            lines.append(f"workspace: {workspace}")
        lines.append("")

        # F33 (UPG-17.1): the leaf name actually matched against the edge/symbol
        # tables — used only for the "any '<leaf>'" caller wording below so it
        # doesn't overstate what was matched when `symbol_name` was qualified.
        class_qualifier, leaf_name = _split_class_qualifier(symbol_name)

        # UPG-TRACE-CALLERS-QUALIFIER-VALIDATION: the qualifier was given but
        # did not resolve to a real definition — lead with the honest miss
        # (and any near-miss suggestion) before any leaf-scoped fallback data
        # below, so the caller LLM never mistakes an unvalidated fallback for
        # a validated answer.
        qualifier_unresolved = bool(trace_result.get("qualifier_unresolved"))
        if qualifier_unresolved:
            lines.append(self._qualifier_unresolved_banner(
                symbol_name, class_qualifier, leaf_name,
                trace_result.get("qualifier_near_miss") or [],
            ))

        # UPG-4.1 / F33: ambiguous symbol OR a resolved "Class.method" query —
        # show callees separated per definition so the LLM sees e.g. resolver
        # `Lock` vs sync `Lock` as distinct, not merged, and a qualified query
        # gets calls scoped to the one class it named rather than every
        # same-leaf definition's calls merged together.
        by_def = trace_result.get("by_definition")
        qualified_class = trace_result.get("qualified_class")
        if by_def and (len(by_def) > 1 or qualified_class):
            by_def_callers = trace_result.get("callers")
            spans, callee_loc = self._trace_call_site_ids(
                workspace, by_def_callers, [entry["callees"] for entry in by_def], chunk_span_lookup,
            )
            if qualified_class:
                lines.append(
                    f"Resolved '{symbol_name}' to {len(by_def)} definition"
                    f"{'s' if len(by_def) != 1 else ''} inside class '{qualified_class}' — "
                    f"calls are shown per definition. (Callers below match the leaf name "
                    f"'{leaf_name}' only and can't be attributed to one class by static "
                    f"analysis.)\n"
                )
            elif qualifier_unresolved:
                lines.append(
                    f"Falling back to every definition of the unscoped leaf name "
                    f"'{leaf_name}' below ({len(by_def)}) — none confirmed to belong "
                    f"to '{class_qualifier}'; calls are shown per definition. "
                    f"(Callers further below match the leaf name only.)\n"
                )
            else:
                lines.append(
                    f"⚠ '{symbol_name}' has {len(by_def)} definitions across modules — "
                    f"calls are shown per definition. (Callers below match the name only "
                    f"and can't be attributed to one definition by static analysis.)\n"
                )
            # UPG-TRACE-CALLER-LIMIT-TRANSPARENCY: the definitions fetch is
            # capped at the same `limit` as every list here, and each rendered
            # definition is a slot against it — when the cap bit, whole
            # definitions (with their calls) are missing. Suppressed for a
            # RESOLVED qualified query: there the fetch ran over ALL classes'
            # definitions and the beyond-cap ones are other-class sites the
            # answer excluded by design, not silently-lost content.
            defs_truncated = int(trace_result.get("definitions_truncated") or 0)
            if defs_truncated > 0 and not qualified_class:
                lines.append(
                    f"(+{defs_truncated} more definition"
                    f"{'s' if defs_truncated != 1 else ''} of '{leaf_name}' beyond "
                    f"the limit not shown — pass a higher limit to show)\n"
                )
            any_callees_found = False
            for entry in by_def:
                d = entry["definition"]
                mod = entry.get("module") or d.file_path
                cs = entry["callees"]
                lines.append(f"[{d.kind}] {d.name} @ {mod}:{d.start_line} — calls ({len(cs)}):")
                if cs:
                    any_callees_found = True
                    for e in cs:
                        lines.append(self._render_callee_line(e, "    ", callee_loc, spans, workspace))
                else:
                    lines.append("    (none found in index)")
                    if d.kind == "class":
                        lines.append(f"    {self._class_trace_note()}")
                note = self._hidden_builtins_note(entry.get("hidden_builtins", 0))
                if note:
                    lines.append(note)
                tnote = self._truncation_note(
                    entry.get("truncated_callees", 0), "callees",
                    raw_capped=entry.get("truncated_callees_raw_capped", False),
                )
                if tnote:
                    lines.append(tnote)
                lines.append("")
            callers = by_def_callers
            callers_empty = callers is not None and not callers
            if callers is not None:
                if callers:
                    lines.append(f"{self._caller_verb(callers)} — any '{leaf_name}' ({len(callers)}):")
                    for e in callers:
                        lines.append(self._render_caller_line(e, spans, workspace))
                    tnote = self._truncation_note(
                        trace_result.get("callers_truncated", 0), "callers",
                        raw_capped=bool(trace_result.get("callers_raw_capped")),
                    )
                    if tnote:
                        lines.append(tnote)
                else:
                    lines.append(f"Called by — any '{leaf_name}': (none found in index)")
            if callers_empty and not any_callees_found:
                lines.append(self._empty_trace_hint(symbol_name))
            return "\n".join(lines)

        # UPG-TRACE-DIRECTION-UNQUERIED-RENDER: `callers`/`callees` are
        # `None` (not `[]`) when the caller did not query that direction —
        # `.get(key)` with no default is what lets the `is None` gate below
        # mean "skip this block entirely" rather than "render a (none found
        # in index) line about a lookup that never happened". `[]` is
        # reserved for "queried and genuinely empty", which still renders
        # its honest miss line.
        callers = trace_result.get("callers")
        callees = trace_result.get("callees")
        if callers is None and callees is None:
            # Neither direction queried — formatter is being called on a
            # result that ran no SQL. Skip the body entirely; the only
            # output is the header above.
            return "\n".join(lines)
        # `_trace_call_site_ids` treats `None` callers as `[]` and an empty
        # `callee_lists` as nothing-to-resolve — passing it the un-queried
        # direction as a literal empty input is safe and keeps the batch
        # span resolver's contract intact.
        spans, callee_loc = self._trace_call_site_ids(
            workspace, callers or [], [callees or []], chunk_span_lookup,
        )
        callers_empty = callers is not None and not callers
        if callers is not None:
            if callers:
                # UPG-TRACE-CALLERS-QUALIFIER-VALIDATION: this is the ONE flat
                # list, always looked up by bare leaf regardless of qualifier
                # (see `trace()`). When the qualifier failed to resolve, label
                # it "any '<leaf>'" — matching the wording the by-definition
                # branch above already uses for the same "can't attribute to
                # one definition" situation — instead of presenting it as the
                # named class's callers.
                if qualifier_unresolved:
                    lines.append(f"{self._caller_verb(callers)} — any '{leaf_name}' ({len(callers)}):")
                else:
                    lines.append(f"{self._caller_verb(callers)} ({len(callers)}):")
                for e in callers:
                    lines.append(self._render_caller_line(e, spans, workspace))
                tnote = self._truncation_note(
                    trace_result.get("callers_truncated", 0), "callers",
                    raw_capped=bool(trace_result.get("callers_raw_capped")),
                )
                if tnote:
                    lines.append(tnote)
            elif qualifier_unresolved:
                lines.append(f"Called by — any '{leaf_name}': (none found in index)")
            else:
                lines.append("Called by: (none found in index)")

        callees_empty = callees is not None and not callees
        if callees is not None:
            if callees:
                # Same flat, bare-leaf lookup and same honesty labeling as
                # callers above — the by-definition breakdown is the only
                # exactly-attributed callees answer; this flat one is not.
                if qualifier_unresolved:
                    lines.append(f"\nCalls — any '{leaf_name}' ({len(callees)}):")
                else:
                    lines.append(f"\nCalls ({len(callees)}):")
                for e in callees:
                    lines.append(self._render_callee_line(e, "  ", callee_loc, spans, workspace))
                tnote = self._truncation_note(
                    trace_result.get("callees_truncated", 0), "callees",
                    raw_capped=bool(trace_result.get("callees_raw_capped")),
                )
                if tnote:
                    lines.append(tnote)
            else:
                if qualifier_unresolved:
                    lines.append(f"\nCalls — any '{leaf_name}': (none found in index)")
                else:
                    lines.append("\nCalls: (none found in index)")
                definitions = trace_result.get("definitions") or []
                if any(d.kind == "class" for d in definitions):
                    lines.append(self._class_trace_note())
            note = self._hidden_builtins_note(trace_result.get("hidden_builtins", 0))
            if note:
                lines.append(note)

        if callers_empty and callees_empty:
            lines.append(self._empty_trace_hint(symbol_name))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_symbol(row: sqlite3.Row) -> Symbol:
        return Symbol(
            symbol_id=row["symbol_id"],
            workspace=row["workspace"],
            name=row["name"],
            kind=row["kind"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            # snippet is populated by locate() after DB fetch — not stored in DB
        )

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> CallEdge:
        return CallEdge(
            from_file=row["from_file"],
            from_symbol=row["from_symbol"],
            from_line=row["from_line"],
            to_symbol=row["to_symbol"],
            edge_type=row["edge_type"],
        )
