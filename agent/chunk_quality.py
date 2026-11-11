"""Language-agnostic chunk-quality heuristics.

The four audited corpora (Python-multi, C, Zig, Rust) all showed the same
failure: low-information chunks outranking real code — bare ``}`` / ``return 0;``
(C), ``const log = std.log;`` (Zig), ``pub use …`` re-export blocks (Rust), and
heading-only markdown. This module centralises the predicates used to:

  * drop / merge trivial chunks at index time            (UPG-1.1)
  * tag re-export / import-only "navigational" chunks    (UPG-1.2)
  * skip vectr-generated config + machine-generated files (UPG-1.3)
  * fold a quality prior into search ranking             (UPG-2.1)
  * de-prioritise test files vs implementation           (UPG-2.3)

Everything here is pure (no I/O) and cheap so it can run per-chunk at both index
and query time.
"""
from __future__ import annotations

import difflib
import re
from pathlib import PurePosixPath

from agent.config import (
    QUALITY_TRIVIAL as _Q_TRIVIAL,
    QUALITY_NAVIGATIONAL as _Q_NAVIGATIONAL,
    QUALITY_NAV_DECLARATION_RESCUE as _Q_NAV_DECLARATION_RESCUE,
    QUALITY_HEADING_ONLY as _Q_HEADING_ONLY,
    QUALITY_GENERATED as _Q_GENERATED,
    QUALITY_VECTR_CONFIG as _Q_VECTR_CONFIG,
    QUALITY_TEST_DEPRIORITISED as _Q_TEST_DEPRIORITISED,
    TEST_FRAMEWORK_FAN_IN_THRESHOLD as _TEST_FRAMEWORK_FAN_IN_THRESHOLD,
    QUALITY_DOC_PROSE as _Q_DOC_PROSE,
    QUALITY_SHORT_PENALTY as _Q_SHORT_PENALTY,
    QUALITY_PRIVATE_SYMBOL as _Q_PRIVATE_SYMBOL,
    TRIVIAL_DOC_MAX_LINES as _TRIVIAL_DOC_MAX_LINES,
    TRIVIAL_ATTR_CLASS_MAX_ATTRS as _TRIVIAL_ATTR_CLASS_MAX_ATTRS,
    INDEXING_BUILD_ARTIFACT_DIR_SUFFIXES as _BUILD_ARTIFACT_DIR_SUFFIXES,
    DUAL_VECTOR_MAX_SIGNATURE_LINES as _DV_MAX_SIGNATURE_LINES,
    DUAL_VECTOR_MAX_DOCSTRING_LINES as _DV_MAX_DOCSTRING_LINES,
    DUAL_VECTOR_MAX_DOCSTRING_CHARS as _DV_MAX_DOCSTRING_CHARS,
    DOCSTRING_DEDUP_LINES as _DOCSTRING_DEDUP_LINES,
    DOCSTRING_DEDUP_MIN_CHARS as _DOCSTRING_DEDUP_MIN_CHARS,
)

# A synthetic node_type stamped on re-export / import-only chunks so the ranker
# can recognise them without re-parsing.
NAVIGATIONAL_NODE_TYPE = "navigational"


# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

# Prefixes the chunker prepends (leading comments, "# class: X" context). These
# are not the chunk's substance, so they're stripped before judging triviality.
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "*/", "<!--", "-->")

# A line that carries no semantic payload on its own.
_TRIVIAL_LINE_RE = re.compile(
    r"""^[\s]*(
        [{}()\[\];,]+                       # pure punctuation: } ; ){ etc.
        | \#\s*(endif|else|elif|pragma\b.*) # C preprocessor noise
        | return(\s+(0|NULL|nullptr|None|true|false|nil|-?\d+))?\s*;?  # bare returns
        | (pass|break|continue|\.\.\.);?    # python/keyword stubs
        | (else|do|try)\s*[:{]?             # lone block openers
    )[\s]*$""",
    re.VERBOSE,
)

# A single-line variable/const declaration. On its own (Zig `variable_declaration`
# nodes, lone module constants) these carry almost no retrieval signal — the audit
# flagged `const log = std.log;`, `const seed = 42;`, `const fs = std.fs;`.
_DECL_LINE_RE = re.compile(r"^[\s]*(pub\s+)?(const|let|var|val|static|final)\s+\w+\s*[:=]")

# A single import / re-export / alias line (navigational, not implementation).
_IMPORT_LINE_RE = re.compile(
    r"""^[\s]*(
        (pub\s+)?use\s                      # rust use / pub use
        | import\s | from\s.+\simport\s     # python / js import
        | export\s+\*                       # js barrel re-export
        | export\s+\{                       # js named re-export
        | (export\s+)?\{?\s*[\w,\s]+\}?\s*from\s  # ts re-export
        | \#include\b                       # C include
        | using\s                           # C++/C# using
        | (const|let|var)\s+\w+\s*=\s*(require\(|@import\() # js require / zig @import alias
        | const\s+\w+\s*=\s*@import\(       # zig @import alias
        | package\s | module\s              # go/zig package decls
    )""",
    re.VERBOSE,
)

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")

# Two-line stub detection (UPG-15.1, Python-focused).
# A chunk whose only content is a declaration header followed by a lone stub body
# (pass / ... / raise NotImplementedError) has no retrieval value.  Structural
# patterns — not tunables.
#
# _STUB_BODY_RE: the second meaningful line is a pure empty-body placeholder.
_STUB_BODY_RE = re.compile(
    r"^[\s]*(pass|\.\.\.|(raise\s+NotImplementedError(\s*\(.*\))?));?[\s]*$"
)

# _DECL_HEADER_RE: the first meaningful line is a class declaration or a
# PARAMETER-LESS function declaration (empty parens, or self/cls only, with an
# optional return-type annotation).  Functions with real parameters
# (e.g. ``def handle(request, uid=None):``) carry semantic signal in their
# signature even when the body is ``pass``, so those are excluded here.
# Class declarations are always matched regardless of base classes, because
# a bare class name + bases without any body is not a useful code chunk.
#
# Matches:
#   class Foo:                class Foo(Base, Meta):
#   def foo():                def foo() -> T:
#   async def foo():          async def foo() -> None:
#   def foo(self):            def foo(cls) -> T:
# Does NOT match (functions with real parameters):
#   def handle(request):      def send(sender, uid=None, **kwargs):
_DECL_HEADER_RE = re.compile(
    r"^[\s]*(?:"
    r"class\s+\w+[^:]*:"                             # class Name (any bases OK) + colon
    r"|(?:async\s+)?def\s+\w+\s*\("                  # def/async def Name(
    r"(?:\s*(?:self|cls)\s*)?"                        # optional self or cls only
    r"\s*\)\s*(?:->[^:]+)?"                          # optional -> return type
    r":"                                              # colon
    r")\s*$"
)

# Attribute-assignment-only class body detection (UPG-15.9 / F25).
# A class header whose body consists ONLY of simple attribute assignments
# (e.g. ``class Meta:\n    model = Writer\n    fields = '__all__'``) is a
# configuration stub with no standalone retrieval value.  Guarded to ONLY fire
# when there is no ``def`` / nested ``class`` / control flow — real small library
# classes (e.g. ``class Meta`` with a custom method) must survive.
#
# _CLASS_HEADER_RE: first meaningful line is a class declaration header.
_CLASS_HEADER_RE = re.compile(r"^[\s]*class\s+\w+[^:]*:\s*$")

# _ATTR_ASSIGN_RE: a line is an attribute assignment of the form
# ``identifier = <value>`` or ``identifier: type = <value>`` (dataclass /
# config-class style).  Matches any assignment; combined with _COMPLEX_RHS_RE to detect
# whether the RHS is a "simple" value (not a function call or dotted access).
_ATTR_ASSIGN_RE = re.compile(r"^[\s]*\w+\s*(?::[^=]+)?\s*=\s*(.+)$")

# _COMPLEX_RHS_RE: the right-hand side of an assignment is "complex" (carries
# real semantic signal) when it contains a function CALL or a dotted ATTRIBUTE
# access.  An assignment like ``username = forms.CharField(...)`` defines a field
# with a specific type — real code with retrieval value.  A "simple" assignment
# like ``model = Writer``, ``fields = '__all__'``, or a bare tuple/list literal
# ``fields = ()`` / ``fields = ('name', 'age')`` is a config option, trivial on
# its own.  Two signals (either makes the RHS complex):
#   * a call:  a name/closing-bracket immediately followed by ``(``
#     (``CharField(``, ``foo()``) — but NOT a grouping/tuple paren ``= (`` where
#     ``(`` is preceded by whitespace/operator, so ``fields = ('a','b')`` stays
#     simple.
#   * dotted attribute access:  an identifier followed by ``.`` (``forms.``,
#     ``models.CASCADE``) — but NOT a numeric literal like ``1.5`` (digit-led).
_COMPLEX_RHS_RE = re.compile(r"[\w\]\)]\s*\(|[A-Za-z_]\w*\.")

# _BARE_CTOR_RHS_RE: the RHS captured by _ATTR_ASSIGN_RE is a bare constructor
# call — a single call to a PascalCase-named callable (the class-naming
# convention shared by Python/JS/TS/Java/C#/Go) with simple arguments only (no
# nested call, no dotted attribute access inside the parens).  This is the
# "declare a module-level instance of an imported type" pattern used by
# re-export/manifest modules (``request_started = Signal()``,
# ``pre_init = ModelSignal(use_caching=True)``) — the statement adds no
# retrieval value beyond the import that already names the type, so a module
# consisting only of imports and such declarations is navigational, not
# implementation (UPG-PREFIX-COMPOSE).
_BARE_CTOR_RHS_RE = re.compile(r"^[A-Z]\w*\([^().]*\)[\s;]*$")

# _DECLARED_NAME_RE: an attribute-assignment line captured as (LHS identifier,
# RHS).  Same shape as _ATTR_ASSIGN_RE but also captures the LHS name, so a
# bare-constructor-manifest line's declared identifier (``request_started`` in
# ``request_started = Signal()``) can be recovered (UPG-NAV-OVERDEMOTE-DECL).
_DECLARED_NAME_RE = re.compile(r"^[\s]*(\w+)\s*(?::[^=]+)?\s*=\s*(.+)$")

# _LEADING_DOCSTRING_DELIM_RE: a line that OPENS a Python triple-quoted string
# at its start (module/file docstring convention).  Used only to recognise a
# LEADING module docstring block so it can be skipped when judging whether the
# rest of a chunk is import-only navigational content — a docstring describes
# the file, it is not itself implementation or an import (UPG-PREFIX-COMPOSE).
_LEADING_DOCSTRING_DELIM_RE = re.compile(r'^("""|\'\'\')')

# Lines that signal real logic in a class body — if any of these appear in the
# body, the chunk is NOT an attribute-only stub.
_HAS_DEF_OR_LOGIC_RE = re.compile(
    r"""^[\s]*(
        (async\s+)?def\s          # method definition
        | class\s                 # nested class
        | (if|elif|else|for|while|try|except|finally|with|raise|return|yield|assert)\b
    )""",
    re.VERBOSE,
)

# _TRIVIAL_ATTR_CLASS_MAX_ATTRS is imported from agent.config (via config.yaml
# ranking.quality_priors.trivial_attr_class_max_attrs).  No inline literal here —
# the config import at the top of this file already binds the name.


def _meaningful_lines(content: str) -> list[str]:
    """Lines that are neither blank nor comment/context-prefix lines."""
    out: list[str] = []
    for raw in content.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(_COMMENT_PREFIXES):
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Chunk predicates
# ---------------------------------------------------------------------------

def is_trivial_chunk(content: str, language: str = "") -> bool:
    """True if the chunk has no standalone retrieval value (UPG-1.1).

    A chunk is trivial when, after dropping comments/context lines, it is empty
    or its only meaningful line is pure punctuation, a bare return, a lone block
    keyword, or a single import. These never answer a query on their own.

    UPG-15.5: HTML/markup and plain-text chunks with ≤ TRIVIAL_DOC_MAX_LINES
    non-blank lines are also trivial.  This catches 1–2-line test-fixture
    templates ("Logged out", "{{ form }}", "<h1>Error</h1>") and egg-info TXT
    stubs (a lone package name, "from-my-custom-list") that otherwise flood short
    natural-language queries.  Multi-line .txt/.rst documentation (a framework's
    docs/howto/*.txt, docs/topics/*.txt) has many more lines and is unaffected.
    """
    raw_nonblank = [l for l in content.splitlines() if l.strip()]
    if not raw_nonblank:
        return True

    # UPG-15.5: language-aware short-prose rule for HTML/TXT doc languages.
    # Checked on raw_nonblank (not _meaningful_lines) so RST heading underlines
    # (===, ---) and HTML tags count as non-blank content.  Any real doc chunk
    # has far more than TRIVIAL_DOC_MAX_LINES non-blank lines; only stub fixtures
    # fall at or below the threshold.
    lang_lower = (language or "").lower()
    if lang_lower in _TRIVIAL_DOC_LANGUAGES and len(raw_nonblank) <= _TRIVIAL_DOC_MAX_LINES:
        return True

    lines = _meaningful_lines(content)
    if not lines:
        # Only comments/headings, no code/text payload. Trivial only when tiny —
        # a large comment/doc block (e.g. a window over an unparsed file) is real
        # coverage and must be kept.
        return len(raw_nonblank) <= 2
    if len(lines) == 1:
        line = lines[0]
        if _TRIVIAL_LINE_RE.match(line) or _IMPORT_LINE_RE.match(line) or _DECL_LINE_RE.match(line):
            return True
        # A single very short token line (e.g. `bad_extension = true;`) with no
        # structure is also low value.
        if len(line) <= 3:
            return True
    if len(lines) == 2:
        # Declaration header + lone stub body → no retrieval value.
        # e.g. "class Style:\n    pass"  or  "def foo():\n    ..."
        # Only fires when: (1) first line is a class/def declaration header,
        # AND (2) second line is a pure stub (pass / ... / raise NotImplementedError).
        # A real second line ("return self.x", "x = 1") is NOT a stub body and
        # keeps the chunk alive.
        if _DECL_HEADER_RE.match(lines[0]) and _STUB_BODY_RE.match(lines[1]):
            return True

    # UPG-15.9 / F25: attribute-assignment-only class body with SIMPLE values.
    # A class whose body is ONLY attribute assignments with no method definitions,
    # nested classes, control flow, OR complex RHS (function calls, dotted access)
    # is a configuration stub (e.g. a web framework's inner ``class Meta:`` with
    # ``model=X, fields='__all__'``).  200+ such 3-line chunks exist in such a
    # framework's test files and flood doc-intent queries with zero educational content.
    #
    # Guard conditions (ALL must hold to classify as trivial):
    #   1. First meaningful line is a class declaration header (not a def).
    #   2. The class body has ≤ _TRIVIAL_ATTR_CLASS_MAX_ATTRS meaningful body lines.
    #   3. Every body line is an attribute assignment (matches _ATTR_ASSIGN_RE).
    #   4. NO body line has a complex RHS: no function call ``(`` or dotted ``.``.
    #      This preserves real form/model classes like
    #      ``username = forms.CharField(...)`` (dotted + parens → complex → kept).
    #   5. NO body line contains a method def, nested class, or control-flow keyword.
    #
    # A class with any method, complex field declaration, or more body lines than
    # _TRIVIAL_ATTR_CLASS_MAX_ATTRS is NOT trivial — it may be a real form,
    # dataclass, NamedTuple, or library class.
    if len(lines) >= 2 and _CLASS_HEADER_RE.match(lines[0]):
        body_lines = lines[1:]
        # Require at least 2 body lines: a single-attr-assignment class like
        # ``class Foo:\n    x = 1`` is kept by the UPG-15.1 invariant (the
        # two-line stub rule only fires for pass/... stubs; a real assignment
        # body keeps the chunk alive).  The UPG-15.9 rule targets the
        # multi-attribute config stubs (class Meta: model=X; fields='__all__').
        if 1 < len(body_lines) <= _TRIVIAL_ATTR_CLASS_MAX_ATTRS:
            # Reject immediately if any line has def/class/control flow
            if not any(_HAS_DEF_OR_LOGIC_RE.match(bl) for bl in body_lines):
                # Accept only if every body line is a simple attribute assignment
                # (RHS has no function call parens or dotted attribute access).
                if all(
                    (m := _ATTR_ASSIGN_RE.match(bl)) and not _COMPLEX_RHS_RE.search(m.group(1))
                    for bl in body_lines
                ):
                    return True

    return False


def _strip_leading_docstring(lines: list[str]) -> list[str]:
    """Drop a leading Python module/file docstring block (UPG-PREFIX-COMPOSE).

    A module docstring (``\"\"\"Multi-consumer multi-producer dispatching
    mechanism...\"\"\"``) describes the file; it is not an import and not
    implementation, so it should not by itself block an otherwise
    import-only re-export shim (e.g. a package ``__init__.py`` that opens
    with a docstring and then re-exports names) from being recognised as
    navigational. Only strips a block that starts at ``lines[0]`` — a
    triple-quoted string appearing later (e.g. inside a function body) is
    left untouched.
    """
    if not lines:
        return lines
    m = _LEADING_DOCSTRING_DELIM_RE.match(lines[0])
    if not m:
        return lines
    delim = m.group(1)
    remainder = lines[0][len(delim):]
    if delim in remainder:
        return lines[1:]
    for i in range(1, len(lines)):
        if delim in lines[i]:
            return lines[i + 1:]
    return lines


def is_navigational_chunk(content: str, language: str = "") -> bool:
    """True if every meaningful line is an import / re-export / alias, or a
    bare module-level instantiation of an imported type (UPG-1.2, extended by
    UPG-PREFIX-COMPOSE).

    These are tables of contents (Rust ``lib.rs`` re-export blocks, Python
    ``__init__`` import aggregators, JS/TS barrels) — they lexically match many
    queries but contain no implementation. The extension also covers Python
    "declaration manifest" modules that do nothing but import a type and
    declare bare module-level instances of it (``request_started = Signal()``)
    — same lack of standalone retrieval value as a pure re-export.
    """
    lines = _meaningful_lines(content)
    if len(lines) < 2:
        # single-line imports are handled by is_trivial_chunk; require a block
        return False
    body = _strip_leading_docstring(lines)
    if not body:
        return False
    nav = 0
    for line in body:
        if _IMPORT_LINE_RE.match(line) or _TRIVIAL_LINE_RE.match(line):
            nav += 1
        elif (m := _ATTR_ASSIGN_RE.match(line)) and _BARE_CTOR_RHS_RE.match(m.group(1).strip()):
            nav += 1
        else:
            return False
    return nav == len(body)


def navigational_declared_identifiers(content: str) -> list[str]:
    """Identifiers declared by bare-constructor-manifest assignment lines.

    A chunk classified navigational via the bare-constructor-manifest pattern
    (``request_started = Signal()``, UPG-PREFIX-COMPOSE) is not a pure
    re-export — the LHS name of each such line IS the corpus-wide unique
    declaration site of that identifier. quality_score() uses this to check
    whether a query lexically names one of the declared identifiers before
    applying the full navigational demotion (UPG-NAV-OVERDEMOTE-DECL / F59):
    a manifest that IS the answer to "where is X defined" should not be
    buried the same way a plain re-export block is. Returns [] for chunks
    with no such lines (pure import blocks keep the full demotion).
    """
    out: list[str] = []
    for line in _meaningful_lines(content):
        m = _DECLARED_NAME_RE.match(line)
        if m and _BARE_CTOR_RHS_RE.match(m.group(2).strip()):
            out.append(m.group(1))
    return out


# Identifier sub-word tokenizer — duplicated (not imported) from
# agent.searcher._code_tokenize's identifier-aware splitting, because
# searcher.py already imports FROM this module (importing back would cycle).
# Used only to compare a chunk's declared identifier names against the
# query's own already-tokenized BM25 tokens (lexical-match-gated), never as a
# standing query-keyword list.
_IDENT_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_IDENT_RE = re.compile(r"([a-z])([A-Z])")


def _identifier_parts(text: str) -> frozenset[str]:
    """Lowercase sub-word tokens of an identifier (camelCase + snake_case split)."""
    expanded = _CAMEL_IDENT_RE.sub(r"\1 \2", text)
    return frozenset(t for t in _IDENT_SPLIT_RE.split(expanded.lower()) if len(t) >= 2)


def is_markdown_heading_only(content: str) -> bool:
    """True if a markdown chunk is essentially just heading(s) + a scrap of body.

    ``_meaningful_lines`` already strips ``#``-prefixed lines (markdown headings
    read as comments), so what's left is the prose body. A chunk with no body, or
    a heading plus only a tiny fragment of text (e.g. ``### vectr_evict_hint`` /
    ``No distinct content again.``), carries no real retrieval signal (UPG-2.1).
    A genuine one-sentence section (≥40 chars of prose) is real content and kept.
    """
    body = _meaningful_lines(content)
    if not body:
        return True
    return len(body) <= 2 and len(" ".join(body)) < 40


# ---------------------------------------------------------------------------
# File predicates
# ---------------------------------------------------------------------------

# vectr's own injected IDE-config files — noise in a *code* search index.
_VECTR_CONFIG_BASENAMES = {
    "claude.md", "agents.md", "gemini.md", "codex.md",
    ".cursorrules", ".mcp.json", "copilot-instructions.md",
}
_VECTR_CONFIG_DIRS = {".cursor", ".vscode"}

# Machine-generated / vendored file patterns (not hand-authored code).
_GENERATED_NAME_RES = [
    re.compile(r".*_db\.h$"),            # generated char-database header, e.g. unicodetype_db.h
    re.compile(r".*_metadata\.h$"),      # generated opcode-metadata header, e.g. pycore_uop_metadata.h
    re.compile(r".*\.pb\.(go|cc|h|py)$"),  # protobuf
    re.compile(r".*_pb2\.pyi?$"),        # python protobuf
    re.compile(r".*\.min\.(js|css)$"),   # minified
    re.compile(r".*\.generated\..*$"),
    re.compile(r".*\.g\.dart$"),
]
_GENERATED_DIR_PARTS = {"clinic", "generated", "__generated__", "gen", "node_modules"}


def is_vectr_config_file(file_path: str) -> bool:
    """True for vectr-injected IDE-config files that should never be indexed (UPG-1.3)."""
    p = PurePosixPath(file_path.replace("\\", "/"))
    if p.name.lower() in _VECTR_CONFIG_BASENAMES:
        return True
    return any(part in _VECTR_CONFIG_DIRS for part in p.parts)


def is_generated_file(file_path: str) -> bool:
    """True for machine-generated / vendored files (UPG-1.3)."""
    p = PurePosixPath(file_path.replace("\\", "/"))
    name = p.name.lower()
    if any(rx.match(name) for rx in _GENERATED_NAME_RES):
        return True
    return any(part.lower() in _GENERATED_DIR_PARTS for part in p.parts)


def is_build_artifact_file(file_path: str) -> bool:
    """True for files inside build-artifact directories (UPG-15.9).

    Detects files whose path contains a directory component that ends with one of
    the configured build-artifact dir suffixes (e.g. ``.egg-info``, ``.dist-info``).
    These directories are entirely machine-generated (Python packaging metadata,
    file-path manifests, PKG-INFO) and contain no educational content — they flood
    BM25 on module/command identifiers.

    Examples that return True:
      ``/project/mypackage.egg-info/SOURCES.txt``
      ``/project/myapp.egg-info/PKG-INFO``
      ``/project/mylib-1.0.dist-info/RECORD``

    Real documentation (``docs/howto/*.txt``) is unaffected — ``docs`` does not
    end with any of the configured suffixes.

    Suffixes are sourced from ``indexing.build_artifact_dir_suffixes`` in
    ``agent/config.yaml`` via ``config.INDEXING_BUILD_ARTIFACT_DIR_SUFFIXES``.
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    # Check every directory component (not the filename itself).
    for part in p.parts[:-1]:
        part_lower = part.lower()
        for suffix in _BUILD_ARTIFACT_DIR_SUFFIXES:
            if part_lower.endswith(suffix):
                return True
    return False


# A C/C++ file basename that starts with "_test" (e.g. "_testutilmodule.c") —
# a shipped internal test module, following the leading-underscore-for-
# internal-module convention some C codebases use for their own test suite
# files (UPG-RUST-DEF-EVICTION / DEF-A). Distinct from the trailing "_test.c"
# / "_test.h" pattern below (a name that ENDS in "_test").
_C_LEADING_TEST_FILE_RE = re.compile(r"^_test.*\.(c|h)$")


def is_test_file(file_path: str) -> bool:
    """True for test files, which should not outrank implementation (UPG-2.3)."""
    p = PurePosixPath(file_path.replace("\\", "/"))
    name = p.name.lower()
    if name.startswith("test_") or name.endswith((
        "_test.py", "_test.go", ".test.ts", ".test.js", ".spec.ts", ".spec.js",
        "_test.c", "_test.h",
    )):
        return True
    if re.match(r"test.*\.(java|kt)$", name):
        return True
    if _C_LEADING_TEST_FILE_RE.match(name):
        return True
    parts = {part.lower() for part in p.parts[:-1]}
    return bool(parts & {"test", "tests", "__tests__", "spec", "testing"})


# ---------------------------------------------------------------------------
# Content-structural inline-test detection (UPG-RUST-DEF-EVICTION / DEF-A)
# ---------------------------------------------------------------------------

# is_test_file() is purely path-based, so test code co-located INSIDE a
# production file evades the demotion above: a Rust #[test] fn (or one inside
# a #[cfg(test)] mod block) living in an otherwise-production .rs file, or a
# Zig inline `test "…" {` block, never has a test-named path to match. These
# are AST-emitted structural markers — the same class of signal as the
# existing _IMPORT_LINE_RE / _TRIVIAL_LINE_RE structural patterns in this
# module (language syntax, not a tunable vocabulary), gated by the chunk's
# own recorded `language` field so a coincidental line shape in an unrelated
# language can never trigger it.

# Rust test-attribute family. The chunker prepends a function's leading
# attribute lines as "comments" (indexer._get_leading_comments treats any
# `#`-prefixed line as a leading-comment line), so a #[test]-attributed fn's
# chunk content already starts with this line today — no reindex required.
_RUST_TEST_ATTR_RE = re.compile(r"^#\[\s*(test|tokio::test|cfg\(test\))\s*\]$")

# Zig inline test-block declaration head: `test "name" {` or bare `test {`.
_ZIG_TEST_DECL_RE = re.compile(r'^test\s*(".*?"\s*)?\{')


def is_content_structural_test_chunk(content: str, language: str = "") -> bool:
    """True if the chunk's own text structurally marks it as test code, even
    when its FILE path is not test-named (UPG-RUST-DEF-EVICTION / DEF-A).

    Scans only the chunk's HEAD region — the leading attribute/comment lines
    the chunker already prepends before the declaration line, plus the
    declaration line itself — and stops at the first line that is neither an
    attribute/comment nor a recognised test-block head. A string literal or
    comment mentioning "#[test]"/"test {" somewhere INSIDE a function body is
    therefore never matched; only a marker that actually precedes/opens the
    declaration counts.

    Known gap (not silently patched — honest limitation of this first cut): a
    helper function defined inside the same `#[cfg(test)] mod { ... }` block
    as a `#[test]` fn, but not itself carrying the `#[test]` attribute, has no
    marker in its own chunk head and is not caught here.
    """
    lang = (language or "").lower()
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if lang == "rust" and _RUST_TEST_ATTR_RE.match(stripped):
            return True
        if lang == "zig" and _ZIG_TEST_DECL_RE.match(stripped):
            return True
        if not (stripped.startswith(_COMMENT_PREFIXES) or stripped.startswith("@")):
            break
    return False


# ---------------------------------------------------------------------------
# Quality prior (UPG-2.1)
# ---------------------------------------------------------------------------

# Documentation languages — prose, not implementation. On code-shaped queries the
# audit found substantive doc prose (blog sections, README walkthroughs, marketing
# HTML) burying real code, because the embedding model scores natural-language prose
# highly against natural-language queries. A mild demotion lets near-tied code edge
# ahead while leaving docs on top when nothing else competes (UPG-2.1).
_DOC_LANGUAGES = {"markdown", "md", "html", "htm", "rst", "text", "txt", "mdx"}

# Languages for which a very short chunk (≤ TRIVIAL_DOC_MAX_LINES non-blank lines)
# is classified as trivial (UPG-15.5). Covers test-fixture HTML templates and
# egg-info / requirements TXT stubs that flood short natural-language queries.
# Markdown is intentionally excluded: is_markdown_heading_only() handles its
# trivial sub-cases already (a 1-line markdown heading is caught there, and a
# single prose sentence has real retrieval value).
_TRIVIAL_DOC_LANGUAGES = {"html", "htm", "text", "txt"}


def is_doc_language(language: str) -> bool:
    """True for documentation/prose languages (vs. implementation code)."""
    return (language or "").lower() in _DOC_LANGUAGES


def is_private_symbol_name(symbol_name: str) -> bool:
    """True for a symbol named by the "internal use" leading-underscore
    convention (a single leading underscore, not a dunder).

    Language-general, not query- or corpus-specific: Python's PEP 8 marks a
    single leading underscore as "internal use only"; the same idiom appears
    across most C-family/JS/Go/Rust codebases for a private helper that
    supports, but is not itself, the public API. Dunder methods (``__init__``,
    ``__str__``, ...) are excluded — they are public protocol hooks, not
    private implementation detail.
    """
    name = symbol_name or ""
    return name.startswith("_") and not name.startswith("__")


# Multipliers applied to the hybrid similarity score — sourced from
# agent/config.yaml (ranking.quality_priors) via agent/config.py.
# The _Q_* aliases are imported at the top of this file so all call sites
# inside this module continue to work without change (UPG-12.1).


def quality_score(
    content: str,
    file_path: str = "",
    language: str = "",
    node_type: str = "",
    query_tokens: frozenset[str] = frozenset(),
    file_fan_in: int = 0,
    symbol_name: str = "",
) -> float:
    """A per-chunk usefulness prior in (0, 1], folded into ranking as a multiplier.

    Relevance × usefulness: similarity already models relevance; this models
    "is this chunk a good answer at all, regardless of similarity". Cheap and
    language-agnostic. Lower = worse answer.

    ``query_tokens`` (already-tokenized BM25 query tokens, not a keyword list)
    softens the navigational demotion when the chunk is a bare-constructor-
    manifest whose declared identifier the query names directly
    (UPG-NAV-OVERDEMOTE-DECL / F59). ``file_fan_in`` (corpus-wide unambiguous
    caller-file count, see symbol_graph.file_fan_in) exempts a shipped
    testing-framework file misclassified by its test-named path from the
    test-file demotion (UPG-TESTPATH-FRAMEWORK-MISCLASS / F58) — the SAME
    exemption also covers a chunk classified as test by its own content
    (is_content_structural_test_chunk, DEF-A) rather than its path.
    ``symbol_name`` (the chunk's bare, unqualified symbol leaf) mildly demotes
    a private/internal helper (UPG-16.1 / F30) — see is_private_symbol_name.
    ``symbol_name`` + ``node_type`` together also EXEMPT a symbol-bearing
    definition chunk from the trivial multiplier (UPG-TRIVIAL-DROP-ALIAS-DEFS)
    — see is_definition_chunk.
    """
    if file_path and is_vectr_config_file(file_path):
        return _Q_VECTR_CONFIG
    if node_type == NAVIGATIONAL_NODE_TYPE or is_navigational_chunk(content, language):
        if query_tokens:
            for name in navigational_declared_identifiers(content):
                parts = _identifier_parts(name)
                if parts and parts <= query_tokens:
                    return _Q_NAV_DECLARATION_RESCUE
        return _Q_NAVIGATIONAL
    # UPG-TRIVIAL-DROP-ALIAS-DEFS: a chunk that is itself a symbol DEFINITION
    # (real symbol_name + a class/struct/enum/interface/type-alias/function/
    # method node_type, is_definition_chunk) is exempt from the trivial
    # multiplier for the same reason it is exempt from the index-time
    # trivial-DROP in _chunking.py's _postprocess_chunks — a one-line alias
    # class (`class ModelForm(Base, metaclass=Meta): pass`) IS the canonical
    # answer to "where is X defined", so scoring it identically to a bare
    # `return`/lone-import chunk buries it beyond what the importance blend
    # can recover. Falls through to the remaining rules below (test/doc/
    # private-symbol/short-chunk demotions all still apply) rather than
    # returning a special-cased score — a definition chunk is scored like
    # any other real chunk, not given a blanket exemption from every prior.
    if is_trivial_chunk(content, language) and not is_definition_chunk(symbol_name, node_type):
        return _Q_TRIVIAL
    if language == "markdown" and is_markdown_heading_only(content):
        return _Q_HEADING_ONLY

    score = 1.0
    if file_path and is_generated_file(file_path):
        score *= _Q_GENERATED
    # DEF-A (UPG-RUST-DEF-EVICTION): a chunk is test code either because its
    # FILE path says so (is_test_file, path-based, UPG-2.3) or because its OWN
    # content structurally marks it as such (a Rust #[test] fn / Zig inline
    # `test "…" {` block living in an otherwise-production file, UPG-2.3's
    # path-only check cannot see these). Same demotion tier and same
    # framework-fan-in escape apply regardless of which signal fired.
    is_test_chunk = (
        (file_path and is_test_file(file_path))
        or is_content_structural_test_chunk(content, language)
    )
    if is_test_chunk and file_fan_in < _TEST_FRAMEWORK_FAN_IN_THRESHOLD:
        score *= _Q_TEST_DEPRIORITISED
    if is_doc_language(language):
        score *= _Q_DOC_PROSE
    if is_private_symbol_name(symbol_name):
        score *= _Q_PRIVATE_SYMBOL

    n_lines = len(_meaningful_lines(content))
    if n_lines <= 2:
        score *= _Q_SHORT_PENALTY
    return score


def quality_demotion_reason(
    content: str,
    file_path: str = "",
    language: str = "",
    node_type: str = "",
    *,
    symbol_name: str = "",
    query_tokens: frozenset[str] = frozenset(),
    file_fan_in: int = 0,
) -> str:
    """Short human label for the dominant quality demotion quality_score applies
    to this chunk, or "" when the chunk carries no demotion (UPG-SCORE-ORDER-
    EXPLAIN). Mirrors quality_score's precedence so a render-time explanation of
    why a higher-relevance chunk ranks below rank 1 names the prior that actually
    demoted it. Presentational only — never affects ranking.

    Reports the single strongest demotion. The exclusive branches (vectr-config,
    navigational, trivial, heading-only) short-circuit exactly as in
    quality_score; the multiplicative demotions (generated/test/doc/private/
    short) are reported in descending severity so the label names the biggest
    single contributor."""
    if file_path and is_vectr_config_file(file_path):
        return "vectr config file"
    if node_type == NAVIGATIONAL_NODE_TYPE or is_navigational_chunk(content, language):
        if query_tokens:
            for name in navigational_declared_identifiers(content):
                parts = _identifier_parts(name)
                if parts and parts <= query_tokens:
                    return ""  # rescued: not demoted for this query
        return "navigational chunk"
    if is_trivial_chunk(content, language) and not is_definition_chunk(symbol_name, node_type):
        return "trivial stub"
    if language == "markdown" and is_markdown_heading_only(content):
        return "heading-only markdown"
    if file_path and is_generated_file(file_path):
        return "generated file"
    is_test_chunk = (
        (file_path and is_test_file(file_path))
        or is_content_structural_test_chunk(content, language)
    )
    if is_test_chunk and file_fan_in < _TEST_FRAMEWORK_FAN_IN_THRESHOLD:
        return "test-file demotion"
    if is_doc_language(language):
        return "docs/prose"
    if is_private_symbol_name(symbol_name):
        return "private helper"
    if len(_meaningful_lines(content)) <= 2:
        return "short/thin chunk"
    return ""


# ---------------------------------------------------------------------------
# Class-context extraction (UPG-F4)
# ---------------------------------------------------------------------------

# Regex to detect a CLASS-prefix line injected by the indexer (UPG-F4).
# The indexer prepends "# class: ClassName\n" to method chunks so the embedding
# has class context.  We extract this at query time to reconstruct the qualified
# name when symbol_name is a bare leaf.
_CLASS_PREFIX_RE = re.compile(r"^#\s*class:\s*(\w+)", re.MULTILINE)


def extract_class_from_content(content: str) -> str:
    """Extract the class name from an indexer-injected '# class: X' prefix line.

    The indexer prepends ``# class: ClassName`` to method chunks so they are
    self-contained for the embedder (indexer.py _collect_chunks_ast).  This
    function recovers that class name at query time so we can reconstruct the
    qualified ``ClassName.leaf`` form when ``symbol_name`` was stored as a bare
    leaf (UPG-F4).

    Returns the class name string, or ``""`` if no prefix is found.
    """
    m = _CLASS_PREFIX_RE.search(content)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Type-definition node_type prior (UPG-RUST-DEF-EVICTION / DEF-B)
# ---------------------------------------------------------------------------

# Tree-sitter node_type strings the indexer actually stamps on a chunk
# (agent/indexer/_chunking.py `_CHUNK_NODE_TYPES` — the node.type recorded on
# CodeChunk, NOT the symbol graph's presentational "kind" label) when the
# chunk is a TYPE DEFINITION (struct/enum/trait/class/interface/typedef)
# rather than a function/method or a type's separate implementation block.
# Verified empirically against the exact `_CHUNK_NODE_TYPES` dict per
# language — REACHABLE means this string can actually be found on an indexed
# chunk's node_type today. All entries below are REACHABLE
# (UPG-RUST-STRUCT-CHUNK-MISSING closed the last DORMANT gaps: Rust
# struct_item/trait_item/enum_item, TypeScript interface_declaration /
# type_alias_declaration / enum_declaration, Go type_declaration, and Java
# interface_declaration / enum_declaration were all added to
# `_CHUNK_NODE_TYPES` so each now gets its own top-level chunk instead of
# being silently swallowed by a sibling chunk or a window fallback):
#   python:      class_definition                    — REACHABLE
#   javascript:  class_declaration                   — REACHABLE
#   typescript:  class_declaration, interface_declaration,
#                type_alias_declaration, enum_declaration — REACHABLE
#   java:        class_declaration, interface_declaration,
#                enum_declaration                    — REACHABLE
#   go:          type_declaration                    — REACHABLE (single node
#                type covers struct/interface/alias forms in this grammar)
#   rust:        struct_item, trait_item, enum_item   — REACHABLE. impl_item
#                is deliberately EXCLUDED from this set: an impl block is an
#                implementation of a type, not the type's own definition
#                (symbol_graph already draws this same distinction, UPG-4.5).
#   c / cpp:     struct_specifier, enum_specifier, type_definition — REACHABLE
#   cpp only:    class_specifier                     — REACHABLE
_TYPE_DEF_NODE_TYPES: frozenset[str] = frozenset({
    "class_definition",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "type_declaration",
    "struct_item",
    "trait_item",
    "enum_item",
    "struct_specifier",
    "enum_specifier",
    "type_definition",
    "class_specifier",
})

# Zig has no distinct type-definition node type: both `pub const Foo = struct
# {...}` (a genuine type definition) and a plain `pub const x = 5;` (an
# ordinary constant) parse to the SAME chunk node_type, "variable_declaration"
# (verified empirically against a live tree-sitter-zig parse and against
# agent/indexer/_chunking.py `_CHUNK_NODE_TYPES["zig"]`). Matching on
# node_type alone would therefore boost every top-level Zig constant, not
# just type factories — so a Zig "variable_declaration" chunk additionally
# needs this narrow, declaration-HEAD-only content check (never the body) for
# the `= struct {` / `= enum {` / `= union {` / `= opaque {` factory shape
# before the type-def prior is granted.
_ZIG_TYPE_FACTORY_HEAD_RE = re.compile(
    r"^(pub\s+)?const\s+\w+\s*=\s*(struct|enum|union|opaque)\b"
)


def is_type_definition_chunk(node_type: str, content: str = "", language: str = "") -> bool:
    """True if a chunk defines a TYPE (struct/enum/trait/class/typedef), as
    opposed to a function/method or a type's separate implementation block
    (UPG-RUST-DEF-EVICTION / DEF-B).

    When a type definition and a usage or test site of the same name compete
    at similar relevance, the definition is the canonical answer to "where is
    X defined" — usages are one vectr_trace call away. This is a chunk-
    PROPERTY check computed only from the chunk's own recorded node_type/
    content/language, never from the query.
    """
    if node_type in _TYPE_DEF_NODE_TYPES:
        return True
    if (language or "").lower() == "zig" and node_type == "variable_declaration":
        first_line = next((l.strip() for l in content.splitlines() if l.strip()), "")
        return bool(_ZIG_TYPE_FACTORY_HEAD_RE.match(first_line))
    return False


# ---------------------------------------------------------------------------
# Module-level function node_type (UPG-SIBLING-TYPEDEF-CROWDING)
# ---------------------------------------------------------------------------

# Tree-sitter node_type strings the indexer stamps on a FUNCTION chunk (see
# agent/indexer/_chunking.py `_CHUNK_NODE_TYPES`) across every language with a
# symbol graph. Deliberately excludes method node_types ("method_definition",
# "method_declaration") — those are always owned by a class/receiver, even on
# the rare occasion the "# class: X" context prefix failed to attach, so they
# must keep going through the existing owning-class attribution path rather
# than being mistaken for a standalone function.
_FUNCTION_NODE_TYPES: frozenset[str] = frozenset({
    "function_definition",   # python, c, cpp
    "function_declaration",  # javascript, typescript, go, zig (go/zig: never a receiver method — see "method_declaration")
    "function_expression",   # javascript, typescript
    "arrow_function",        # javascript, typescript
    "function_item",         # rust (outside an impl_item)
})

# Method node_type strings the indexer stamps on a chunk (see `_CHUNK_NODE_TYPES`
# in agent/indexer/_chunking.py) for languages whose grammar gives a method its
# own distinct node type — javascript/typescript ("method_definition"), go/java
# ("method_declaration"). Python has no separate method node type: a method's
# `def` parses to the same "function_definition" already in
# `_FUNCTION_NODE_TYPES` above, distinguished from a module-level function only
# by its (indentation-derived) class context, not by node_type. Kept apart
# from `_FUNCTION_NODE_TYPES` because that set's own docstring records a
# DELIBERATE exclusion of method node types for a different purpose (owning-
# class importance attribution, UPG-SIBLING-TYPEDEF-CROWDING) — this set is
# for the orthogonal question of "is this chunk a symbol DEFINITION at all"
# (UPG-TRIVIAL-DROP-ALIAS-DEFS), where a method is exactly as much a
# definition as a module-level function.
_METHOD_NODE_TYPES: frozenset[str] = frozenset({
    "method_definition",
    "method_declaration",
    # UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE: Java constructors are chunked
    # (node_type "constructor_declaration", added to `_CHUNK_NODE_TYPES["java"]`)
    # and are as much a symbol definition as any method — a one-line constructor
    # deserves the same trivial-drop exemption.
    "constructor_declaration",
})

# The full symbol-DEFINITION node_type family (UPG-TRIVIAL-DROP-ALIAS-DEFS):
# type definitions (class/struct/enum/interface/trait/type-alias) plus
# function/method definitions. A chunk whose node_type is in this set actually
# DECLARES a named symbol — as opposed to a navigational/window/markdown-
# section chunk, or an implementation block (Rust `impl_item`) that is not
# itself the type's own definition site.
_DEFINITION_NODE_TYPES: frozenset[str] = (
    _TYPE_DEF_NODE_TYPES | _FUNCTION_NODE_TYPES | _METHOD_NODE_TYPES
)


def is_definition_chunk(symbol_name: str, node_type: str) -> bool:
    """True if a chunk is a real symbol DEFINITION — a named class, struct,
    enum, interface, type-alias, function, or method (UPG-TRIVIAL-DROP-ALIAS-
    DEFS).

    Used to EXEMPT an otherwise content-trivial chunk (e.g. a one-line
    ``class ModelForm(BaseModelForm, metaclass=ModelFormMetaclass): pass``
    alias, or a single-statement function body) from the UPG-1.1 trivial-drop
    at index time: a symbol-bearing definition is the canonical answer to
    "where is X defined" even when its body is a bare ``pass``/one-liner, so
    dropping it makes the symbol structurally unfindable by search (the
    symbol graph / ``locate`` is unaffected — this is a search-only gap).
    A pure chunk-PROPERTY check (node_type + the chunk's own recorded
    symbol_name), never the query.
    """
    return bool(symbol_name) and node_type in _DEFINITION_NODE_TYPES


def is_module_level_function_chunk(node_type: str, class_ctx: str) -> bool:
    """True if a chunk defines a MODULE-LEVEL function (not a method), as
    opposed to a method chunk whose owning class happened to resolve (or a
    type definition / type's implementation block).

    A module-level function has its own reference-frequency importance (ARCH-2
    extension, UPG-SIBLING-TYPEDEF-CROWDING) exactly like a class/struct name
    does — a corpus-central function (e.g. one call thousands of sites route
    through) can otherwise lose to a same-file, rarely-referenced sibling
    function at a near-tie base relevance, the same crowding class of problem
    ARCH-2 already fixes for classes.

    `class_ctx` is the caller's already-extracted
    `extract_class_from_content(content)` result — passed in rather than
    recomputed here since every caller already has it at hand. This is a pure
    chunk-PROPERTY check (node_type + the chunk's own recovered class context),
    never the query.
    """
    return node_type in _FUNCTION_NODE_TYPES and not class_ctx


# ---------------------------------------------------------------------------
# Purpose-text distillation (ARCH-4 dual-vector pool entry)
# ---------------------------------------------------------------------------

# node_types the chunker stamps that are never symbol definitions — a purpose
# vector (qualified signature + docstring) only makes sense for a chunk that
# actually declares a function/method/class/type. Markdown sections, sliding-
# window fallback chunks, and re-export blocks carry no signature to distil.
_NON_SYMBOL_NODE_TYPES = {NAVIGATIONAL_NODE_TYPE, "window", "section"}

# The "# class: X" context line the indexer prepends to method chunks (see
# extract_class_from_content) — excluded from the leading-doc scan below so it
# doesn't get embedded twice (once as the qualified name, once as raw text).
_CLASS_PREFIX_LINE_RE = re.compile(r"^#\s*class:\s*\w+\s*$")

# A declaration line's block-opening terminator: python's trailing ':' or a
# C-family/Rust/Go/Java/Zig trailing '{'. Trailing whitespace/comment-safe.
_SIGNATURE_END_RE = re.compile(r"[:{]\s*(//.*|#.*)?$")

# First statement of a Python function/class body is a string literal — the
# docstring convention. Matches from the very start of the (stripped) body
# text; DOTALL so a multi-line triple-quoted docstring is captured whole.
_PY_DOCSTRING_RE = re.compile(
    r'^[rRbBuU]{0,2}(?P<q>"""|\'\'\')(?P<body>.*?)(?P=q)', re.DOTALL,
)
# One-line plain-quoted docstring (less common but valid Python).
_PY_DOCSTRING_ONELINE_RE = re.compile(
    r"^[rRbBuU]{0,2}(?P<q>['\"])(?P<body>(?:(?!(?P=q)).)*)(?P=q)\s*$"
)


def is_symbol_bearing_chunk(symbol_name: str, node_type: str) -> bool:
    """True if a chunk declares a real symbol worth a purpose vector.

    A symbol chunk has a non-empty `symbol_name` AND a node_type that is an
    actual AST definition node — not a navigational/window/markdown-section
    chunk (those have no signature to distil).
    """
    return bool(symbol_name) and node_type not in _NON_SYMBOL_NODE_TYPES


def _leading_doc_and_code(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a chunk's lines into (leading doc/decorator lines, remaining code).

    "Leading" comments/decorators (JSDoc, rustdoc, godoc, `@decorator`) precede
    the declaration for most languages — already prepended to chunk content by
    the chunker's `_get_leading_comments`. The injected "# class: X" context
    line is skipped (it is not documentation prose).
    """
    doc: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _CLASS_PREFIX_LINE_RE.match(stripped):
            i += 1
            continue
        if stripped.startswith(_COMMENT_PREFIXES) or stripped.startswith("@"):
            doc.append(stripped)
            i += 1
            continue
        break
    return doc, lines[i:]


def _extract_signature(code_lines: list[str]) -> tuple[list[str], int]:
    """Return (signature lines, index of first body line) from a declaration.

    Accumulates lines from the start of `code_lines` until one ends the block
    opener (python ':' / brace-family '{'), or `_DV_MAX_SIGNATURE_LINES` is
    reached — bounds pathological multi-line parameter lists.
    """
    sig: list[str] = []
    for i, line in enumerate(code_lines[:_DV_MAX_SIGNATURE_LINES]):
        sig.append(line.strip())
        if _SIGNATURE_END_RE.search(line.rstrip()):
            return sig, i + 1
    return sig, min(len(code_lines), _DV_MAX_SIGNATURE_LINES)


def _first_paragraph(text: str) -> str:
    """The text up to (not including) the first blank line, else the whole text.

    PEP 257 convention (mirrored by Google/NumPy docstring styles): a
    multi-line docstring is a one-line summary, a blank line, then an
    elaborated description — often a structured block (``Args:``, attribute
    lists, examples). The summary line alone already carries the purpose;
    everything after the first blank line is detail for a human reader, not
    additional intent signal. Keeping it anyway measurably dilutes the
    embedding (see ARCH-4-DEBUG spike evidence), the same class of problem the
    purpose vector exists to defeat — just recurring one level down inside the
    docstring itself for structured multi-paragraph text.
    """
    m = re.search(r"\n[ \t]*\n", text)
    return text[: m.start()] if m else text


def _extract_python_docstring(body_lines: list[str]) -> str:
    """First-statement docstring from a Python function/class body, if any."""
    body_text = "\n".join(body_lines).strip()
    if not body_text:
        return ""
    m = _PY_DOCSTRING_RE.match(body_text) or _PY_DOCSTRING_ONELINE_RE.match(
        body_text.splitlines()[0].strip() if body_text.splitlines() else ""
    )
    if not m:
        return ""
    doc = _first_paragraph(m.group("body").strip())
    doc_lines = doc.splitlines()[:_DV_MAX_DOCSTRING_LINES]
    return "\n".join(doc_lines)[:_DV_MAX_DOCSTRING_CHARS]


def build_purpose_text(
    content: str, symbol_name: str, node_type: str, language: str = "",
) -> str | None:
    """Distil a symbol-bearing chunk down to qualified signature + docstring.

    ARCH-4: the STEP-0 spike proved a mechanical implementation body dilutes
    the intent-bearing tokens (signature + docstring) when mean-pooled into a
    single body embedding — the canonical chunk can miss dense pool entry
    entirely even though its own docstring already paraphrases the query. This
    builds the body-stripped text embedded as the chunk's second "purpose"
    vector: `ClassName.symbol_name` (class-qualified when the indexer recorded
    class context) + the raw declaration line(s) (which carry the parameter
    list) + the docstring (Python: first body statement) or leading
    comment/decorator block (other languages' pre-declaration doc convention).

    Returns None for non-symbol chunks (`is_symbol_bearing_chunk` False) — no
    purpose vector is stored for markdown/navigational/window chunks. An
    undocumented symbol still returns a non-None signature-only text (graceful
    degradation — no docstring found is not an error).
    """
    if not is_symbol_bearing_chunk(symbol_name, node_type):
        return None

    lines = content.splitlines()
    class_ctx = extract_class_from_content(content)
    qualified_name = f"{class_ctx}.{symbol_name}" if class_ctx else symbol_name

    leading_doc, code_lines = _leading_doc_and_code(lines)
    signature, body_start = _extract_signature(code_lines)

    docstring = ""
    if (language or "").lower() == "python":
        docstring = _extract_python_docstring(code_lines[body_start:])

    parts = [qualified_name]
    if signature:
        parts.append("\n".join(signature))
    if leading_doc:
        # Same size caps as the Python docstring branch (max_docstring_lines /
        # max_docstring_chars) — a long JSDoc/rustdoc/godoc header block dilutes
        # the purpose embedding exactly like an untruncated Python docstring
        # would; this branch was previously uncapped.
        capped_doc = leading_doc[:_DV_MAX_DOCSTRING_LINES]
        parts.append("\n".join(capped_doc)[:_DV_MAX_DOCSTRING_CHARS])
    if docstring:
        parts.append(docstring)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dedup (UPG-2.2)
# ---------------------------------------------------------------------------

def normalized_content(content: str) -> str:
    """Whitespace-collapsed lowercase form for exact/near-duplicate detection."""
    return re.sub(r"\s+", " ", content).strip().lower()


# A leading attribute/decorator line — Python/Java `@decorator`/`@Override`,
# Rust/Zig `#[attr]` — is a STRUCTURAL marker, not documentation prose, even
# though `_leading_doc_and_code` sweeps it into the same "leading comment"
# block as a real doc-comment (both start with `@`/`#`, the same convention
# `_get_leading_comments` uses at chunk-creation time). Left unfiltered here,
# it shadows the real docstring in two ways: (1) a chunk whose ONLY leading
# line is a decorator (e.g. `@abc.abstractmethod` immediately above a Python
# method whose docstring is its FIRST body statement, not a leading comment)
# makes `doc_text` non-empty, so the Python-docstring fallback below never
# runs and the dedup key becomes the decorator text itself (too short/generic
# to mean anything, or worse: identical across UNRELATED overrides that merely
# share `@abc.abstractmethod`); (2) two DIFFERENT Rust structs sharing only a
# `#[derive(Debug, Clone)]` line with no `///` doc would wrongly key on that
# shared attribute and collapse together. Filtering these lines out before
# keying — and falling through to the doc that follows them, or to no key at
# all if there is none — fixes both.
_ATTR_DECORATOR_LINE_RE = re.compile(r"^(@\w|#\[)")


def leading_docstring_key(content: str, language: str = "") -> str:
    """Normalized leading docstring/comment-block key for near-dup collapse
    (UPG-RUST-DEF-EVICTION / DEF-C).

    Byte-identical full-content dedup (`normalized_content`, UPG-2.2) misses
    near-duplicate boilerplate: several thin wrapper/test chunks that copy the
    same rustdoc/JSDoc/docstring header onto otherwise-different bodies still
    bury the canonical definition under look-alike results. This returns a
    second, independent dedup key built ONLY from the chunk's own leading
    doc/comment lines (reusing the same `_leading_doc_and_code` /
    `_extract_python_docstring` split used for purpose-text distillation),
    with attribute/decorator lines filtered out (see `_ATTR_DECORATOR_LINE_RE`
    above), capped at `_DOCSTRING_DEDUP_LINES` lines.

    Returns "" (never a valid dedup key — the caller must treat an empty
    string as "do not collapse this chunk on this key") when the chunk has no
    leading doc at all (once attribute/decorator lines are filtered out), or
    when the normalized header is shorter than `_DOCSTRING_DEDUP_MIN_CHARS` —
    a trivial/near-empty header must not fold together chunks that merely
    share a blank or one-word comment (or, per the above, a bare decorator).
    """
    lines = content.splitlines()
    leading_doc, code_lines = _leading_doc_and_code(lines)
    leading_doc = [l for l in leading_doc if not _ATTR_DECORATOR_LINE_RE.match(l)]

    doc_text = "\n".join(leading_doc)
    if not doc_text and (language or "").lower() == "python":
        _, body_start = _extract_signature(code_lines)
        doc_text = _extract_python_docstring(code_lines[body_start:])

    if not doc_text:
        return ""

    capped = "\n".join(doc_text.splitlines()[:_DOCSTRING_DEDUP_LINES])
    normalized = normalized_content(capped)
    if len(normalized) < _DOCSTRING_DEDUP_MIN_CHARS:
        return ""
    return normalized


def _strip_leading_docstring_block(content: str, language: str = "") -> str:
    """Return the chunk's BODY text with the leading docstring/comment block
    removed (LANE-NEARDUP-DEDUP body-similarity gate).

    The LANE-NEARDUP-DEDUP content-only near-duplicate key needs to compare
    BODIES rather than whole chunks, because the case it exists for —
    leading docstrings differ, bodies are near-verbatim — is precisely
    the case where including the docstring in the comparison drags
    SequenceMatcher.ratio below threshold. This function is the inverse
    of `leading_docstring_key`: same docstring-detection machinery
    (`_leading_doc_and_code` + the same attribute/decorator filter +
    the Python first-statement-docstring fallback), but it returns the
    body span instead of a normalized key.

    For a chunk with NO leading doc (no pre-declaration comment block
    AND no Python first-statement docstring), the body IS the content —
    the helper returns the original content unchanged. That subset
    behaves exactly as the previous whole-content comparison did; only
    the docstring-bearing subset changes behaviour.

    Reusing `leading_docstring_key`'s machinery rather than writing a
    second docstring parser guarantees the two stay in lockstep: a chunk
    that `leading_docstring_key` returns "" for (no doc, or too short)
    also gets a no-op strip here, and a chunk whose docstring is
    detected as a particular span by the key is stripped of the same
    span by this helper. No second source of truth.
    """
    lines = content.splitlines()
    leading_doc, code_lines = _leading_doc_and_code(lines)
    # Same attribute/decorator filter as leading_docstring_key: a leading
    # `@decorator` / `#[derive(...)]` line is structural metadata, not
    # documentation, and is excluded from the "doc" portion so the doc
    # that follows it (or the absence of one) is what the body comparison
    # gets to ignore. The decorator itself lands in `leading_doc` here
    # but is dropped from the returned "doc" portion — the body span
    # (code_lines) is unchanged either way and is what we return.
    leading_doc = [l for l in leading_doc if not _ATTR_DECORATOR_LINE_RE.match(l)]

    if leading_doc:
        # Body is everything after the pre-declaration comment block.
        return "\n".join(code_lines)

    # No pre-declaration comment block. For Python, fall through to the
    # first-statement docstring (the chunk opens with the def line and
    # the docstring is the first body statement). This mirrors the
    # fallback in leading_docstring_key exactly.
    if (language or "").lower() == "python":
        _, body_start = _extract_signature(code_lines)
        body_after_sig = code_lines[body_start:]
        if body_after_sig:
            joined = "\n".join(body_after_sig)
            # _PY_DOCSTRING_RE is multi-line (DOTALL); _PY_DOCSTRING_ONELINE_RE
            # handles the single-line plain-quoted case. Same pair
            # leading_docstring_key's Python fallback ultimately relies on
            # via _extract_python_docstring, just here we're matching against
            # the raw body to recover the SPAN of the docstring instead of
            # its text.
            m = _PY_DOCSTRING_RE.match(joined) or _PY_DOCSTRING_ONELINE_RE.match(
                body_after_sig[0].strip()
            )
            if m:
                # The docstring may span multiple lines; figure out how many
                # body lines it consumed so the body we return skips them.
                doc_end = m.end()
                lines_consumed = joined[:doc_end].count("\n") + 1
                remaining = body_after_sig[lines_consumed:]
                return "\n".join(code_lines[:body_start] + remaining)

    # No docstring found — body is the whole content (no-op strip).
    return content


# ASCII letters, digits, and underscore — the characters that can fill a
# template slot in identifier-or-literal position. Operators, parens, dots,
# quotes, and spaces are not token chars. Defined as a module-level frozenset
# so the per-opcode membership test below stays a constant-time lookup rather
# than a per-character regex or repeated str method call.
_TOKEN_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_"
)


def _is_token_span(s: str) -> bool:
    """True if `s` is non-empty and consists solely of ASCII letters, digits,
    and underscores.

    Used by `_is_templated_body_difference` to classify each
    SequenceMatcher opcode's differing spans: an opcode whose two sides
    are both token-only is a candidate "template slot" diff (an
    identifier rename, a type rename, a string-literal-content
    change, or a digit index). An opcode with any non-token char on
    either side is a structural change that disqualifies the whole
    diff from the templated shape.

    String literal contents like `foo` / `bar` pass naturally:
    SequenceMatcher emits one replace per differing char (e.g. `f`
    vs `b`), and each single-char span is alphanumeric. The
    surrounding `"` chars are matched as equal opcodes and never
    reach this check.
    """
    return bool(s) and all(ch in _TOKEN_CHARS for ch in s)


# One identifier-like token or one bare number. Masking both to the same
# placeholder is what makes _is_templated_body_difference independent of how
# difflib chooses to align two bodies of differing length.
_TEMPLATE_SLOT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _is_templated_body_difference(a: str, b: str) -> bool:
    """True when the differences between two chunks' BODIES are concentrated
    in token-like values (identifiers, type names, string-literal contents,
    embedded numeric indices) — the known false-collapse shape for the
    LANE-NEARDUP-DEDUP content-only key (LANE-TEMPLATED-BODY-GUARD brief).

    Bodies that are identical except for an embedded slot value
    (``accessor_0``/``accessor_1``, ``compute_0``/``compute_1``,
    ``get_name``/``get_email``, ``&str``/``&Path``, ``"foo"``/``"bar"``) are
    extremely common in generated clients, dispatch tables, field accessors,
    enum handlers and parameterised tests. After `normalized_content`
    collapses whitespace, such bodies differ in only a handful of characters
    and clear the 0.95 content-similarity gate while representing genuinely
    distinct symbols. Without a guard, the content-only key would collapse
    them — silently destroying correct results, which is strictly worse than
    the wasted result-list slot the key exists to reduce.

    Rule: every non-equal SequenceMatcher opcode must have BOTH of its
    spans consist solely of ASCII letters, digits, and underscores (a
    "token-only" span, per `_is_token_span`). A single opcode with a
    non-token char on either side — an operator swap, a paren change, a
    dot for a method call, a comment difference, a string literal with
    embedded punctuation — disqualifies the whole diff as templated.
    This is the strict, conservative direction the brief calls for:
    when in doubt, refuse to flag as templated (and let the
    SequenceMatcher ratio gate decide on its own merits). A false
    positive here is a wasted result-list slot; a false negative is a
    real answer silently deleted from the result set. The costs are
    not symmetric.

    The widening from the previous digit-only rule (more than 50%% of
    differing chars are digits) is what the brief explicitly asks
    for: real templated code also varies by identifier, by type, and
    by string literal. The previous rule caught only the digit-index
    shape (e.g. ``accessor_0`` against ``accessor_1``) and let every
    other templated shape through to the content-similarity gate.

    Boundary case just outside this rule (read it to judge the line):
    bodies that differ in tokens AND in a non-token char. Example —
    ``compute(x): return x * 2`` against ``compute(x): return x * 3.0``.
    The ``2`` is a digit (token-only) but the ``3.0`` contains a ``.``
    (non-token), so the whole diff is NOT flagged as templated even
    though the function is clearly a parameterized shape. A slightly
    wider rule (allow ``.`` inside numeric literals) would flag this
    as templated; the current rule chooses not to, because the
    ``* 2`` vs ``* 3.0`` shape is a real semantic difference
    (integer vs float multiplication), not a templated slot fill, and
    letting the SequenceMatcher ratio gate decide is the safer
    default.

    Does NOT catch:
      * bodies that differ in a non-token char (operator swap, paren
        change, dot for a method call);
      * bodies that differ in a comment (comments contain non-token
        chars like ``/``, ``*``, and spaces);
      * bodies that differ in a string literal with embedded
        punctuation (``"hello, world"`` has a comma and a space);
      * bodies that differ purely in a single very long identifier
        rename whose SequenceMatcher ratio is still ≥ 0.95 — a
        pathological case the strict all-token rule catches
        regardless, since the diff is still all alphanumeric. If
        the diff is so large that the SequenceMatcher ratio falls
        below 0.95, the content-similarity gate already prevents the
        collapse and the guard is irrelevant.

    Cheap: one SequenceMatcher.get_opcodes() pass on already-stripped
    bodies (bodies are at most a few hundred chars and the
    LANE-NEARDUP-DEDUP key already runs SequenceMatcher on them) and
    a constant-time set-membership test per non-equal opcode. The
    guard is a property of the two bodies' diff structure, not a
    query-side classification. Fed into `_is_content_near_duplicate`
    (agent/searcher.py) as a third gate alongside the structural-
    node_type and content-similarity gates.
    """
    if a == b:
        return False
    # autojunk=False is REQUIRED, not a tuning choice. difflib's default
    # autojunk heuristic treats any element appearing in more than 1%% of a
    # sequence of 200+ elements as junk and excludes it from matching. These
    # are CHARACTER sequences and a code chunk is routinely longer than 200
    # characters, so ordinary letters get classed as junk and the ratio
    # collapses. Measured on two Rust bodies differing by a single word:
    # 0.158 with the default against 0.995 with autojunk off. Every
    # similarity threshold in this module is meaningless without it.
    # ALIGNMENT-INDEPENDENT by construction. The previous formulation walked
    # SequenceMatcher opcodes and required every non-equal span to be
    # token-only, rejecting any pure insert or delete. That only ever worked
    # for SAME-LENGTH renames: changing `name` to `phone` is 4 chars against
    # 5, so the aligner must emit an insert, and the rule rejected the very
    # shape it exists to catch. Measured on one fixture, it flagged
    # name/email but missed name/phone, name/address, email/phone,
    # email/address and phone/address.
    #
    # Instead, mask every identifier-like token and every bare number to a
    # single placeholder and compare the results. Two bodies are the same
    # shape with different slot values exactly when their masked forms are
    # equal while the originals differ. No alignment is involved, so token
    # length cannot change the answer.
    masked_a = _TEMPLATE_SLOT_RE.sub("\x00", a)
    masked_b = _TEMPLATE_SLOT_RE.sub("\x00", b)
    # a == b was excluded above, so equal masked forms mean the difference
    # lives entirely in slot values. Unequal masked forms mean the bodies
    # differ in structure (an operator, a bracket, a comment, punctuation),
    # which is a real difference and the similarity gate's business.
    return masked_a == masked_b
