"""AST-aware code chunking: tree-sitter parsers, chunk collection, fallback windows."""
from __future__ import annotations

import re
from pathlib import Path

from agent.chunk_quality import (
    NAVIGATIONAL_NODE_TYPE,
    is_definition_chunk,
    is_navigational_chunk,
    is_trivial_chunk,
)
from agent.config import (
    INDEXING_MAX_CHUNK_LINES as _MAX_CHUNK_LINES,
    INDEXING_MAX_CHUNK_CHARS as _MAX_CHUNK_CHARS,
    INDEXING_CLASS_HEADER_LINES as _CLASS_HEADER_LINES,
    INDEXING_FLOW_SCAN_HEAD_BYTES as _FLOW_SCAN_HEAD_BYTES,
    INDEXING_FLOW_PRAGMA as _FLOW_PRAGMA,
    INDEXING_FLOW_SECONDARY_MARKERS as _FLOW_SECONDARY_MARKERS,
)
from agent.indexer._constants import LANG_BY_EXT
from agent.indexer._types import CodeChunk


# ---------------------------------------------------------------------------
# Tree-sitter AST chunker
# ---------------------------------------------------------------------------

_PARSER_CACHE: dict[str, object] = {}


def _get_parser(language: str):
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]
    try:
        from tree_sitter import Language, Parser
        if language == "python":
            import tree_sitter_python as ts_lang
        elif language == "javascript":
            import tree_sitter_javascript as ts_lang
        elif language == "typescript":
            import tree_sitter_typescript as ts_lang
            parser = Parser(Language(ts_lang.language_typescript()))
            _PARSER_CACHE[language] = parser
            return parser
        elif language == "tsx":
            # UPG-JSFLOW-SYMBOLS: used to parse Flow-typed `.js` — tsx is the
            # widest JS-family grammar (JSX + type syntax) so it also covers
            # Flow-typed React components, not just plain type annotations.
            import tree_sitter_typescript as ts_lang
            parser = Parser(Language(ts_lang.language_tsx()))
            _PARSER_CACHE[language] = parser
            return parser
        elif language == "go":
            import tree_sitter_go as ts_lang
        elif language == "rust":
            import tree_sitter_rust as ts_lang
        elif language == "java":
            import tree_sitter_java as ts_lang
        elif language == "zig":
            import tree_sitter_zig as ts_lang
        elif language == "c":
            import tree_sitter_c as ts_lang
        elif language == "cpp":
            import tree_sitter_cpp as ts_lang
        else:
            return None
        parser = Parser(Language(ts_lang.language()))
        _PARSER_CACHE[language] = parser
        return parser
    except Exception:
        return None


# UPG-JSFLOW-SYMBOLS: grammar used to parse Flow-typed `.js` — see `_get_parser`
# and `_parser_language_for` below.
_FLOW_JS_GRAMMAR = "tsx"


def is_flow_javascript(code: str) -> bool:
    """Cheap Flow-type-syntax detector for `.js` source.

    tree-sitter-javascript treats Flow syntax (`@flow` pragma, `import type
    {...}`, `: Type` annotations, generics) as ERROR nodes, which desyncs the
    symbol walk — canonical functions go missing and keyword tokens can be
    misattributed as symbol names (UPG-JSFLOW-SYMBOLS). Detection scans only
    the first `INDEXING_FLOW_SCAN_HEAD_BYTES` bytes — O(1) per file, checked
    once at parse time, never per AST node.
    """
    head = code[:_FLOW_SCAN_HEAD_BYTES]
    if _FLOW_PRAGMA in head:
        return True
    return any(marker in head for marker in _FLOW_SECONDARY_MARKERS)


def _parser_language_for(language: str, code: str) -> str:
    """Grammar key to parse `code` with — may differ from `language`, the
    extension-derived key used for chunk/symbol node-type lookups (which stays
    unchanged so results still land under the "javascript" bucket). A `.js`
    file signalling Flow parses with the typescript/tsx grammar instead of the
    plain javascript grammar (UPG-JSFLOW-SYMBOLS); every other language is a
    no-op passthrough. Falls back to `language` itself if grammar loading
    later fails (handled by `_get_parser`'s own try/except).
    """
    if language == "javascript" and is_flow_javascript(code):
        return _FLOW_JS_GRAMMAR
    return language


# UPG-12.1: _MAX_CHUNK_LINES / _CLASS_HEADER_LINES are sourced from
# agent/config.yaml (indexing.*) via agent/config.py — imported above as
# _MAX_CHUNK_LINES / _CLASS_HEADER_LINES.  The alias names are kept so all
# existing call sites work without change.

# Node types that represent class declarations (handled specially — emit header + recurse).
# This is a fast-path/safety net: every one of these already recurses correctly
# today, and is kept as an explicit set so that guarantee never depends on the
# structural check below. Any OTHER container-shaped node type (an impl/extension
# block, a namespace, an interface/trait with default methods, …) is detected
# structurally by `_is_member_container` instead of being enumerated here —
# see UPG-IMPL-BLOCK-CHUNK-TRUNCATION.
_CLASS_NODE_TYPES = {"class_definition", "class_declaration"}

# Node types that represent top-level code units worth indexing per language
#
# UPG-RUST-STRUCT-CHUNK-MISSING: a type DEFINITION (struct/trait/enum/interface/
# type-alias) is a standalone top-level unit exactly like a function or class —
# omitting it from this set means it is either silently swallowed into whatever
# sibling chunk happens to claim the file, or the whole file falls back to one
# anonymous "window" chunk if no other target type is present. Each language's
# type-def node type is added here once verified (via a live tree-sitter parse)
# to be a real standalone top-level node in that grammar, activating
# chunk_quality._TYPE_DEF_NODE_TYPES / is_type_definition_chunk (DEF-B) for it:
#   rust:       struct_item / trait_item / enum_item — verified top-level nodes
#               distinct from impl_item (an implementation of a type, not its
#               definition — UPG-4.5 draws the same line in the symbol graph).
#   typescript: interface_declaration / type_alias_declaration / enum_declaration
#               — each a standalone top-level type definition (verified).
#   go:         type_declaration — a single node type covers struct/interface/
#               alias forms in this grammar (verified); the type's name is
#               nested one level under a `type_spec` child, handled below in
#               `_extract_symbol_name`.
#   java:       interface_declaration / enum_declaration — standalone top-level
#               definitions alongside the already-chunked class_declaration.
# zig `test_declaration` (`test "name" { ... }`) is added too, but for a
# different reason: it is not a type definition, it activates the DORMANT
# Zig inline-test detection in chunk_quality.is_content_structural_test_chunk
# (DEF-A) — without its own chunk, a Zig test block is invisible to the index
# rather than present-but-demoted, which is a worse outcome (it can still
# surface via the window fallback, undemoted, on a file with no functions).
_CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "function_expression", "arrow_function", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "function_expression", "arrow_function", "class_declaration",
                   "method_definition", "interface_declaration", "type_alias_declaration", "enum_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item", "enum_item"},
    # UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE: constructor_declaration joins the
    # other Java member declarations — a constructor is a definition a caller
    # can search for, exactly like the methods already listed here.
    "java": {"method_declaration", "constructor_declaration", "class_declaration",
             "interface_declaration", "enum_declaration"},
    "zig": {"function_declaration", "variable_declaration", "test_declaration"},
    "c": {"function_definition", "struct_specifier", "enum_specifier", "type_definition"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier",
            "enum_specifier", "type_definition", "namespace_definition"},
}

_SYMBOL_FIELD: dict[str, str] = {
    "python": "name",
    "javascript": "name",
    "typescript": "name",
    "go": "name",
    "rust": "name",
    "java": "name",
    "zig": "name",
}


# C/C++ symbol-name extraction (shared with symbol_graph). Needed because C nests
# the name under the declarator chain: in `PyObject *PyDict_New(void)` the only
# direct `type_identifier` child is the RETURN type (PyObject), not the name.
_C_TYPE_NAME_NODES = {"struct_specifier", "union_specifier", "enum_specifier", "class_specifier"}


def _c_declarator_name(node, code_bytes: bytes) -> str:
    """Follow a C/C++ declarator chain (pointer/function/array/parenthesized) to the name."""
    cur = node
    for _ in range(12):  # bounded — declarator nesting is shallow
        if cur is None:
            return ""
        t = cur.type
        if t in ("identifier", "field_identifier", "type_identifier"):
            return code_bytes[cur.start_byte:cur.end_byte].decode("utf-8", errors="replace")
        if t == "qualified_identifier":  # C++ Foo::bar → the trailing member name
            last = None
            for c in cur.named_children:
                if c.type in ("identifier", "field_identifier", "destructor_name", "operator_name"):
                    last = c
            return code_bytes[last.start_byte:last.end_byte].decode("utf-8", errors="replace") if last else ""
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            nxt = next((c for c in cur.named_children
                        if c.type.endswith("declarator") or c.type in ("identifier", "qualified_identifier")), None)
        cur = nxt
    return ""


def c_symbol_name(node, code_bytes: bytes) -> str:
    """Name of a C/C++ symbol-defining node (handles return-type-vs-name confusion)."""
    t = node.type
    if t in _C_TYPE_NAME_NODES or t == "namespace_definition":
        nm = node.child_by_field_name("name")
        return code_bytes[nm.start_byte:nm.end_byte].decode("utf-8", errors="replace") if nm is not None else ""
    if t in ("preproc_def", "preproc_function_def"):
        nm = node.child_by_field_name("name")
        return code_bytes[nm.start_byte:nm.end_byte].decode("utf-8", errors="replace") if nm is not None else ""
    # function_definition / type_definition / declaration → walk the declarator chain
    d = node.child_by_field_name("declarator")
    return _c_declarator_name(d if d is not None else node, code_bytes)


def _extract_symbol_name(node, language: str, code_bytes: bytes) -> str:
    if language in ("c", "cpp"):
        nm = c_symbol_name(node, code_bytes)
        if nm:
            return nm
    # Go nests a type_declaration's own name one level down, under a type_spec
    # child (`type Point struct {...}` -> type_declaration -> type_spec ->
    # name field), rather than exposing it as a direct child like every other
    # node type this function handles (verified via a live tree-sitter-go
    # parse — UPG-RUST-STRUCT-CHUNK-MISSING).
    if language == "go" and node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                nm = child.child_by_field_name("name")
                if nm is not None:
                    return code_bytes[nm.start_byte:nm.end_byte].decode("utf-8", errors="replace")
        return ""
    # UPG-RUST-IMPL-SYMBOL-NAME-TRAIT-VS-TYPE: an `impl_item` exposes no `name`
    # field, so resolution fell through to the positional child-scan below,
    # which takes the FIRST identifier-ish child in document order — the TRAIT
    # for `impl Trait for Type { }`. On the chunk side that mislabeled
    # symbol_name AND, worse, fed `class_context`: every member chunk of a
    # trait impl carried `# class: <trait>` in its embedding text. Resolve via
    # the grammar's `type` field (present on both inherent and trait impls),
    # matching symbol_graph._get_symbol_name and _graph._impl_owner_type_name;
    # unwrap generic wrappers bounded, "" rather than falling back to the scan.
    if language == "rust" and node.type == "impl_item":
        cur = node.child_by_field_name("type")
        for _ in range(6):  # bounded — generic/reference nesting is shallow
            if cur is None:
                return ""
            if cur.type == "type_identifier":
                return code_bytes[cur.start_byte:cur.end_byte].decode("utf-8", errors="replace")
            cur = cur.child_by_field_name("type")
        return ""
    # Prefer tree-sitter's explicit `name` field when the grammar exposes one:
    # the positional child-scan below returns the FIRST identifier-ish child,
    # which for a Java `method_declaration` with a non-primitive return type is
    # the RETURN TYPE (`Producer foo()` → `type_identifier` "Producer" precedes
    # `identifier` "foo"), naming the chunk after its return type
    # (UPG-JAVA-METHOD-NAME-EXTRACTION). The scan stays the fallback for node
    # types the grammar doesn't expose a `name` field for.
    nm = node.child_by_field_name("name")
    if nm is not None:
        return code_bytes[nm.start_byte:nm.end_byte].decode("utf-8", errors="replace")
    for child in node.children:
        # "type_identifier" covers a type definition's own name in grammars
        # that distinguish it from a value identifier — e.g. a TS/Rust
        # struct/interface/type-alias/class name (verified via live parses;
        # UPG-RUST-STRUCT-CHUNK-MISSING). Checked in the same pass as
        # "identifier" so declaration order still decides which child wins
        # when both could appear (a function's own name always precedes any
        # nested return-type token).
        if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
            return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return ""


def _get_leading_comments(lines: list[str], start_line: int) -> str:
    """Return the comment/decorator block immediately preceding start_line (1-indexed)."""
    collected: list[str] = []
    i = start_line - 2  # 0-indexed line just above the node
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith(("#", "//", "*", "/**", "@")):
            collected.insert(0, lines[i])
            i -= 1
        elif not stripped:
            i -= 1  # skip blank separator lines
        else:
            break
    return "\n".join(collected)


# Node types that exist PURELY as a member-group wrapper directly inside a
# container's `body` field — never a chunk-worthy node in their own right, and
# never a general statement/expression wrapper, only a grammar artifact that
# groups a fixed run of members together as one intermediate node. A wrapper
# type belongs here only when it is verified (via a live tree-sitter parse) to
# be nested exactly one level between `body` and the real member declarations,
# with nothing else ever occurring at that position.
#
# UPG-JAVA-ENUM-METHOD-CHUNK-DEPTH: tree-sitter-java nests an enum's methods as
# enum_declaration -> [body] enum_body -> enum_body_declarations ->
# method_declaration — one hop deeper than a class's methods (class_declaration
# -> [body] class_body -> method_declaration directly), because the grammar
# groups the `;` separator and every member after it under one node. This is
# NOT the same shape as a JS closure/anonymous-class false positive: those are
# reachable only through an intervening statement/expression node (a distinct,
# open-ended node family with no fixed member list), which is why a general
# depth increase would reintroduce that class of bug — this allowlist instead
# names the exact, closed set of grammar wrapper node types verified to hold
# nothing but members, and `_is_member_container` may descend through one of
# them exactly one extra hop, never further.
_MEMBER_WRAPPER_NODE_TYPES = frozenset({"enum_body_declarations"})


def _is_member_container(node, target_types: set[str]) -> bool:
    """True when `node` is a CONTAINER — its own body directly lists (or lists
    via exactly one `_MEMBER_WRAPPER_NODE_TYPES` hop) at least one further
    chunk-worthy member — rather than a leaf definition.

    UPG-IMPL-BLOCK-CHUNK-TRUNCATION: `_CLASS_NODE_TYPES` alone under-recognizes
    containers — a Rust `impl_item`, a C++ `namespace_definition`/`class_specifier`,
    a Java `interface_declaration` with default methods, a Swift-style
    `extension_declaration` are all containers too, and enumerating every such
    node type per language is exactly the kind of special-casing that doesn't
    generalize (and was already caught missing `impl_item`, `class_specifier`,
    and `namespace_definition` — none of which are Rust-specific). Detected
    structurally instead: does this node's tree-sitter `body` field directly
    list (one hop down) another node whose type is itself in `target_types`?

    Deliberately checks only DIRECT children of the body field, not the whole
    subtree — a member declaration is always a direct child of its container's
    body list in every grammar checked (tree-sitter python/rust/java/cpp), while
    a chunk-worthy node reachable only several hops deeper — a JS callback
    passed as a call argument, an anonymous inner class built inside a method
    body — is always wrapped in at least one intervening statement/expression
    node first. The one-hop rule is what tells a real member apart from an
    incidental nested occurrence, without naming any language's node types.

    UPG-JAVA-ENUM-METHOD-CHUNK-DEPTH: a body-field child that is itself a known
    member-group WRAPPER (`_MEMBER_WRAPPER_NODE_TYPES`) is allowed exactly one
    further hop — its own children are checked the same way direct body
    children are. This does not touch the JS-closure protection above: the
    wrapper allowlist is a closed, verified set of grammar node types, never
    the open-ended statement/expression families a closure is wrapped in.
    """
    body = node.child_by_field_name("body")
    if body is None:
        return False
    for child in body.children:
        if child.type in target_types:
            return True
        if child.type in _MEMBER_WRAPPER_NODE_TYPES:
            if any(grandchild.type in target_types for grandchild in child.children):
                return True
    return False


def _emit_overflow_chunks(
    results: list[CodeChunk],
    node,
    lines: list[str],
    file_path: str,
    language: str,
    cap: int,
    prefix_line_count: int,
    covered_spans: list[tuple[int, int]],
    class_context: str,
) -> None:
    """Re-emit a truncated node's post-cap remainder as bounded continuation
    chunks (UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE).

    The header cap exists so one giant container can't flood the index with a
    single enormous chunk — that trade is kept untouched. But capping a chunk
    while nothing re-emits what it dropped made content past the cap
    structurally unreachable: not in any chunk (search can't find it), not a
    symbol (the symbol graph has no field/constructor/enum-constant entries).
    A Java enum whose constants ARE its meaning lost everything past the cap.

    This restores coverage without reopening the flood the cap prevents:

    - The kept chunk is emitted FIRST, byte-for-byte unchanged (same id,
      same capped content, same full-span metadata) — this function only ever
      ADDS chunks.
    - Only the UNCOVERED remainder is re-emitted: line ranges past what the
      kept chunk shows (`cap` lines minus the comment/context prefix it
      consumed) that no member chunk emitted by the recursion already
      carries. Member chunks' spans are extended upward across the leading
      comment/blank lines `_get_leading_comments` prepended into their own
      content, so a doc comment is never emitted twice — once inside its
      member's chunk and once as orphan glue.
    - Runs containing no non-blank line are skipped (pure whitespace between
      members); each surviving run is sliced into disjoint ≤_MAX_CHUNK_LINES
      pieces with HONEST sub-spans — unlike the header chunk's
      full-span-id-with-capped-content quirk, every source line a piece's span
      names appears in it verbatim and complete (one synthetic `# class:`
      context line may precede them, mirroring the chunks being continued), so
      the `path:start-end` ids never point at text the reader can't see.

    Cost: only nodes whose content exceeded their cap gain chunks, at
    ceil(uncovered_lines / _MAX_CHUNK_LINES) per uncovered run — a 500-member
    enum (~550 lines, leaf-capped at 150) yields 3 continuation chunks where
    previously ~400 lines simply vanished; a typical class whose every member
    got its own chunk yields nothing retrievable (blank-line gaps are filtered
    here; any lone closing-brace residue is content-trivial with no symbol
    name, so the existing UPG-1.1 trivial-drop pass removes it).

    node_type "overflow" deliberately matches none of chunk_quality's
    definition/type-def/function sets, so continuation chunks get neutral
    ranking priors — they must never be mistaken for the canonical definition
    site of the container whose tail they carry.
    """
    start_line = node.start_point[0] + 1  # 1-indexed
    end_line = node.end_point[0] + 1
    # Last file line the kept (capped) chunk actually shows: its first
    # `cap` content lines include `prefix_line_count` leading-comment/context
    # lines consumed before the node's own text begins.
    visible_last = start_line + cap - 1 - prefix_line_count
    region_lo = max(visible_last + 1, start_line)
    if region_lo > end_line:
        return

    # Extend each member span upward over the leading comment/blank lines its
    # own chunk already carries (same prefixes _get_leading_comments collects),
    # then merge into a union of already-covered ranges.
    covered: list[list[int]] = []
    for ms, me in sorted(covered_spans):
        ext = ms
        i = ms - 2  # 0-indexed line just above the member's first
        while i >= 0:
            stripped = lines[i].strip()
            if stripped and not stripped.startswith(("#", "//", "/*", "*", "@")):
                break
            ext = i + 1
            i -= 1
        if covered and ext <= covered[-1][1] + 1:
            covered[-1][1] = max(covered[-1][1], me)
        else:
            covered.append([ext, me])

    # Complement of the covered union within [region_lo, end_line].
    runs: list[tuple[int, int]] = []
    cur = region_lo
    for s, e in covered:
        if e < cur:
            continue
        if s > cur:
            runs.append((cur, min(s - 1, end_line)))
        cur = e + 1
        if cur > end_line:
            break
    if cur <= end_line:
        runs.append((cur, end_line))

    context_prefix = f"# class: {class_context}\n" if class_context else ""
    for r0, r1 in runs:
        if r1 < r0 or not any(l.strip() for l in lines[r0 - 1:r1]):
            continue  # pure whitespace between members — nobody's content
        for ps in range(r0, r1 + 1, _MAX_CHUNK_LINES):
            pe = min(ps + _MAX_CHUNK_LINES - 1, r1)
            results.append(CodeChunk(
                chunk_id=f"{file_path}:{ps}-{pe}",
                content=context_prefix + "\n".join(lines[ps - 1:pe]),
                file_path=file_path,
                language=language,
                node_type="overflow",
                start_line=ps,
                end_line=pe,
                symbol_name="",
            ))


def _collect_chunks_ast(
    node,
    code_bytes: bytes,
    lines: list[str],
    language: str,
    file_path: str,
    target_types: set[str],
    results: list[CodeChunk],
    class_context: str = "",
) -> None:
    if node.type in target_types:
        start = node.start_point[0]  # 0-indexed
        end = node.end_point[0]
        raw = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        symbol = _extract_symbol_name(node, language, code_bytes)

        # Prepend leading comments/decorators (stripped from AST node but semantically important)
        leading = _get_leading_comments(lines, start + 1)

        # Prepend class context so method chunks are self-contained for the embedder
        context_prefix = f"# class: {class_context}\n" if class_context else ""

        parts = [p for p in [leading, context_prefix + raw] if p]
        content = "\n".join(parts)

        # Cap very long chunks — container bodies (classes, impl/extension blocks,
        # namespaces, …) can be thousands of lines. A CONTAINER only keeps its
        # header here; its members are emitted as their own chunks below, and
        # whatever the header dropped that NO member carries (fields, initializers,
        # enum constants, …) is re-emitted by _emit_overflow_chunks below, so
        # nothing past the cap is lost (UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE).
        # A leaf definition (a plain function, a struct with no nested members)
        # keeps the full budget instead, since nothing else will carry the rest
        # of its content; if even that budget truncates it, its tail is re-emitted
        # by the same overflow path.
        is_container = node.type in _CLASS_NODE_TYPES or _is_member_container(node, target_types)
        cap = _CLASS_HEADER_LINES if is_container else _MAX_CHUNK_LINES
        content_lines = content.splitlines()
        truncated = len(content_lines) > cap
        if truncated:
            content = "\n".join(content_lines[:cap])

        chunk_id = f"{file_path}:{start + 1}-{end + 1}"
        results.append(CodeChunk(
            chunk_id=chunk_id,
            content=content,
            file_path=file_path,
            language=language,
            node_type=node.type,
            start_line=start + 1,
            end_line=end + 1,
            symbol_name=symbol,
        ))

        member_spans: list[tuple[int, int]] = []
        if is_container:
            # Also recurse into the body so members get their own chunks with context.
            # This walks every child unconditionally (via the plain fallback loop
            # below, for any non-target-type node) all the way down to the real
            # members, so a member-group WRAPPER between `body` and its members
            # (UPG-JAVA-ENUM-METHOD-CHUNK-DEPTH — e.g. Java's enum_body_declarations)
            # is traversed through automatically once `is_container` is True for
            # the enclosing node; no separate wrapper-aware traversal is needed here.
            emitted_before = len(results)
            for child in node.children:
                _collect_chunks_ast(child, code_bytes, lines, language, file_path,
                                    target_types, results, class_context=symbol)
            member_spans = [(c.start_line, c.end_line) for c in results[emitted_before:]]
        if truncated:
            # The cap silently dropped this node's tail. Re-emit only what no
            # chunk emitted above already carries — never the kept chunk's own
            # bytes a second time, and never a member's doc comment twice.
            # The tail CONTINUES existing chunks, so it inherits their context
            # line: a container's leftover fields/constants continue the
            # member set ("# class: <Container>", like every member chunk);
            # a leaf's own tail continues the leaf's chunk (same class_context
            # the kept chunk used — usually none for a top-level definition).
            # Never the node's OWN name: "# class: big_fn" on big_fn's tail
            # would claim the function is an enclosing class.
            prefix_count = (len(leading.splitlines()) if leading else 0) + (1 if context_prefix else 0)
            _emit_overflow_chunks(
                results=results,
                node=node,
                lines=lines,
                file_path=file_path,
                language=language,
                cap=cap,
                prefix_line_count=prefix_count,
                covered_spans=member_spans,
                class_context=symbol if is_container else class_context,
            )
        return  # don't recurse further for non-container nodes (avoids duplicate nested defs)

    for child in node.children:
        _collect_chunks_ast(child, code_bytes, lines, language, file_path,
                            target_types, results, class_context=class_context)


def _fallback_window_chunks(lines: list[str], file_path: str, language: str) -> list[CodeChunk]:
    """Sliding-window chunker for files with no tree-sitter grammar.

    Window=200, overlap=50:
    - 200 lines captures ~3-5 typical functions worth of context — large enough for
      a coherent embedding, small enough that unrelated code doesn't dilute it.
      (100 lines is too small for classes; 500 lines degrades embedding quality.)
    - 50-line overlap ensures a function starting near the tail of one window is
      fully present in the next window, so it's never split across chunk boundaries.
    """
    window, overlap = 200, 50
    chunks: list[CodeChunk] = []
    i = 0
    while i < len(lines):
        end = min(i + window, len(lines))
        content = "\n".join(lines[i:end])
        chunk_id = f"{file_path}:{i + 1}-{end}"
        chunks.append(CodeChunk(
            chunk_id=chunk_id,
            content=content,
            file_path=file_path,
            language=language,
            node_type="window",
            start_line=i + 1,
            end_line=end,
            symbol_name="",
        ))
        i += window - overlap
    return chunks


def _chunk_markdown(lines: list[str], file_path: str) -> list[CodeChunk]:
    """Split markdown at heading boundaries for coherent section-level embeddings.

    Each ATX heading (# through ######) starts a new chunk. Content before the
    first heading is treated as a preamble section. Falls back to window chunks
    if the file has no headings at all (e.g. a flat paragraph document).
    """
    chunks: list[CodeChunk] = []
    current_lines: list[str] = []
    current_start = 1
    current_symbol = ""

    for i, line in enumerate(lines, 1):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append(CodeChunk(
                    chunk_id=f"{file_path}:{current_start}-{i - 1}",
                    content=content,
                    file_path=file_path,
                    language="markdown",
                    node_type="section",
                    start_line=current_start,
                    end_line=i - 1,
                    symbol_name=current_symbol,
                ))
            current_start = i
            current_symbol = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(CodeChunk(
                chunk_id=f"{file_path}:{current_start}-{len(lines)}",
                content=content,
                file_path=file_path,
                language="markdown",
                node_type="section",
                start_line=current_start,
                end_line=len(lines),
                symbol_name=current_symbol,
            ))

    return chunks or _fallback_window_chunks(lines, file_path, "markdown")


def _is_rst_adornment_line(line: str) -> bool:
    """True when `line` is a reStructuredText section-adornment line: a single
    repeated punctuation character and nothing else (the docutils convention
    an RST title/section overline or underline follows — ``===...===``,
    ``---...---``, ``~~~...~~~``; docutils does not restrict the character
    set, any repeated non-alphanumeric, non-whitespace character qualifies).
    Purely structural — the same repetition test the docutils reference
    parser itself applies, not a curated character/keyword list.
    """
    s = line.strip()
    if not s:
        return False
    ch = s[0]
    if ch.isalnum() or ch.isspace():
        return False
    return all(c == ch for c in s)


def _rst_section_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Locate reStructuredText setext-style section headings: a non-blank
    title line immediately followed by an adornment line at least as long as
    the title (docutils' section-title rule), with an optional matching
    overline of the same character and length directly above the title
    absorbed into the same heading block rather than misread as a second
    heading of its own.

    Returns ``(block_start, underline_idx, title)`` 0-indexed triples in
    document order — ``block_start`` is the title line, or the overline's
    line when a matching overline is present.
    """
    n = len(lines)
    headings: list[tuple[int, int, str]] = []
    i = 1
    while i < n:
        if _is_rst_adornment_line(lines[i]):
            title_idx = i - 1
            title = lines[title_idx].strip()
            if title and not _is_rst_adornment_line(lines[title_idx]) and len(title) <= len(lines[i].strip()):
                block_start = title_idx
                if title_idx > 0 and _is_rst_adornment_line(lines[title_idx - 1]):
                    over = lines[title_idx - 1].strip()
                    under = lines[i].strip()
                    if over[0] == under[0] and len(over) == len(under):
                        block_start = title_idx - 1
                headings.append((block_start, i, title))
        i += 1
    return headings


def _chunk_prose_headings(lines: list[str], file_path: str, language: str) -> list[CodeChunk]:
    """Split .txt/.rst prose at reStructuredText setext-style section headings
    — the same per-section splitting `_chunk_markdown` does for the ATX (`#`)
    heading convention, applied to prose docs that use RST's title-plus-
    underline convention instead (UPG-TXT-CHUNK-COVERAGE). Structural only:
    every file is split identically regardless of what will later be searched
    for.

    Neither "txt" nor "rst" has a tree-sitter grammar, so a multi-hundred-line
    doc previously fell straight to `_fallback_window_chunks` — 200-line/
    50-overlap windows sized for 3-5 coherent CODE functions, not prose. That
    merges every unrelated subsection within a 200-line span into one diluted
    chunk instead of giving each documented topic (and its own heading text,
    e.g. a doc's title) its own retrievable chunk. Returns `[]` — triggering
    the caller's window-chunk fallback, matching `_chunk_markdown`'s own
    fallback pattern — when the file has no RST heading structure at all
    (a flat README.txt/LICENSE.txt with no section markers).
    """
    headings = _rst_section_headings(lines)
    if not headings:
        return []

    chunks: list[CodeChunk] = []
    first_start = headings[0][0]
    if first_start > 0:
        preamble = "\n".join(lines[:first_start]).strip()
        if preamble:
            chunks.append(CodeChunk(
                chunk_id=f"{file_path}:1-{first_start}",
                content=preamble,
                file_path=file_path,
                language=language,
                node_type="section",
                start_line=1,
                end_line=first_start,
                symbol_name="",
            ))

    n = len(lines)
    for idx, (block_start, _underline_idx, title) in enumerate(headings):
        content_end = headings[idx + 1][0] if idx + 1 < len(headings) else n
        block_lines = lines[block_start:content_end]
        # Cap each section at the same budget a code container's body is
        # capped at (_MAX_CHUNK_LINES, indexing.max_chunk_lines) — a doc
        # section with no further nested heading can otherwise run to
        # 1000+ lines, producing an outlier chunk whose embedding cost
        # dwarfs every normal chunk sharing its batch (measured on the real
        # django corpus: an uncapped 1191-line section blew a batch encode
        # past available memory). Sub-splitting on the same cap — rather
        # than truncating like `_collect_chunks_ast` does for an oversized
        # code container — keeps the whole section retrievable instead of
        # silently losing its tail; only oversized SECTIONS are affected,
        # a normal heading's content still becomes exactly one chunk.
        for sub_start in range(0, len(block_lines), _MAX_CHUNK_LINES):
            sub_lines = block_lines[sub_start:sub_start + _MAX_CHUNK_LINES]
            content = "\n".join(sub_lines).strip()
            if not content:
                continue
            chunk_start = block_start + sub_start
            chunk_end = min(chunk_start + len(sub_lines), content_end)
            chunks.append(CodeChunk(
                chunk_id=f"{file_path}:{chunk_start + 1}-{chunk_end}",
                content=content,
                file_path=file_path,
                language=language,
                node_type="section",
                start_line=chunk_start + 1,
                end_line=chunk_end,
                symbol_name=title,
            ))
    return chunks


def _split_oversized_chunk(c: CodeChunk) -> list[CodeChunk]:
    """Sub-split one chunk whose content exceeds `_MAX_CHUNK_CHARS` into pieces
    each within the cap (UPG-WINDOW-CHUNK-BYTE-CAP). The caller has already
    decided this chunk is worth keeping — pieces are never re-judged
    individually, so none of them can be silently dropped and lose the tail.

    Two carving strategies, chosen per chunk:

    Line-granular — when the content is verbatim file lines 1:1 with the
    chunk's recorded `[start_line, end_line]` span (window fallback chunks,
    plain AST definitions), the same way `_chunk_prose_headings` sub-splits
    an oversized section: accumulate whole lines up to the cap, emit, repeat.
    Every full-line piece keeps honest line spans and a natural
    `path:start-end` id. A single line longer than the cap itself (the
    reproduced shape: a giant numpy array dump inside an otherwise small
    .txt file, or any minified/generated one-liner — the window path is
    shared by EVERY language's fallback) is hard-split into cap-char runs;
    each run's span is that single line, so its id carries a `#<k>` ordinal
    suffix to stay unique (`path:L-L#0`, `path:L-L#1`, …). Natural ids never
    contain '#', so these can't collide with any other chunk; every
    downstream consumer tolerates them (`vectr_fetch` resolves ids by exact
    match; only the misalignment "nearest ids" ranking skips ids whose range
    doesn't parse as two ints, which is acceptable degradation for
    pathological input).

    Character-granular — when the mapping between content lines and the
    recorded span is broken (prose/markdown section chunks are stripped;
    AST container-header chunks keep only their header lines but record the
    FULL container span; AST definition chunks prepend leading comments),
    slicing at line boundaries would assign spans that lie. The content is
    instead carved into cap-char runs that all inherit the parent's own
    `[start_line, end_line]`, each with a `#<k>`-suffixed id for uniqueness.

    Unlike the window walker's 50-line overlap, sub-split pieces are DISJOINT:
    overlap would multiply the very byte cost this cap exists to bound, and
    the prose sub-split precedent doesn't overlap either.
    """
    content_lines = c.content.split("\n")
    span_lines = c.end_line - c.start_line + 1
    one_to_one = (
        c.start_line > 0
        and c.end_line >= c.start_line
        and len(content_lines) == span_lines
    )

    def _piece(content: str, start_line: int, end_line: int, k: int | None) -> CodeChunk:
        suffix = "" if k is None else f"#{k}"
        return CodeChunk(
            chunk_id=f"{c.file_path}:{start_line}-{end_line}{suffix}",
            content=content,
            file_path=c.file_path,
            language=c.language,
            node_type=c.node_type,
            start_line=start_line,
            end_line=end_line,
            symbol_name=c.symbol_name,
        )

    pieces: list[CodeChunk] = []
    if one_to_one:
        acc: list[str] = []
        acc_chars = 0
        acc_start = c.start_line

        def flush(last_line: int) -> None:
            nonlocal acc, acc_chars
            if acc:
                pieces.append(_piece("\n".join(acc), acc_start, last_line, None))
                acc = []
                acc_chars = 0

        for offset, line in enumerate(content_lines):
            file_line = c.start_line + offset
            if len(line) > _MAX_CHUNK_CHARS:
                flush(file_line - 1)
                for k in range(0, -(-len(line) // _MAX_CHUNK_CHARS)):
                    pieces.append(_piece(
                        line[k * _MAX_CHUNK_CHARS:(k + 1) * _MAX_CHUNK_CHARS],
                        file_line, file_line, k,
                    ))
                acc_start = file_line + 1
                continue
            added = len(line) + (1 if acc else 0)  # +1 for the join newline
            if acc and acc_chars + added > _MAX_CHUNK_CHARS:
                flush(file_line - 1)
                acc_start = file_line
                added = len(line)
            acc.append(line)
            acc_chars += added
        flush(c.end_line)
    else:
        for k in range(0, -(-len(c.content) // _MAX_CHUNK_CHARS)):
            pieces.append(_piece(
                c.content[k * _MAX_CHUNK_CHARS:(k + 1) * _MAX_CHUNK_CHARS],
                c.start_line, c.end_line, k,
            ))
    return pieces


def _postprocess_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    """Wave 1 chunk hygiene applied to every chunker path.

    - UPG-1.1: drop standalone trivial chunks (bare punctuation/return, lone
      import/const) — they have no retrieval value and flood top-N on ties.
      EXEMPT (UPG-TRIVIAL-DROP-ALIAS-DEFS): a chunk that is itself a symbol
      DEFINITION (`is_definition_chunk` — real `symbol_name` + a class/struct/
      enum/interface/type-alias/function/method node_type) is kept even when
      its content alone would read as trivial. A one-line alias like
      ``class ModelForm(BaseModelForm, metaclass=ModelFormMetaclass): pass``
      IS the canonical answer to "where is X defined" — dropping it makes the
      symbol structurally unfindable by search, even though the symbol graph
      / `locate` sees it fine. `is_trivial_chunk` itself is unchanged (other
      callers still judge these chunks' content-only quality at rank time);
      only the DROP decision is gated on the chunk's own recorded properties.
    - UPG-1.2: tag re-export / import-only blocks as navigational so the ranker
      can heavily down-weight them (they're a table of contents, not an answer).
    - UPG-WINDOW-CHUNK-BYTE-CAP: a kept chunk whose content exceeds
      `_MAX_CHUNK_CHARS` (a single long line defeats every LINE-based cap on
      the window fallback, which is shared by every language) is sub-split
      into bounded pieces by `_split_oversized_chunk`. The split runs AFTER
      the keep/drop/tag decisions above and its pieces are never re-judged:
      a hard-split run of one huge data-dump line must not be misread by
      `is_trivial_chunk`'s short-doc rule (a ≤2-non-blank-line .txt stub) and
      dropped — that would silently discard exactly the tail bytes this split
      exists to make retrievable.
    """
    out: list[CodeChunk] = []
    for c in chunks:
        if is_trivial_chunk(c.content, c.language) and not is_definition_chunk(c.symbol_name, c.node_type):
            continue
        if c.node_type != NAVIGATIONAL_NODE_TYPE and is_navigational_chunk(c.content, c.language):
            c.node_type = NAVIGATIONAL_NODE_TYPE
        if len(c.content) > _MAX_CHUNK_CHARS:
            out.extend(_split_oversized_chunk(c))
        else:
            out.append(c)
    return out


def chunk_file(file_path: str) -> list[CodeChunk]:
    """Parse a file and return AST-aware chunks (falls back to windows)."""
    path = Path(file_path)
    ext = path.suffix.lower()
    language = LANG_BY_EXT.get(ext, "")

    try:
        code = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = code.splitlines()
    if not lines:
        return []

    if language == "markdown":
        return _postprocess_chunks(_chunk_markdown(lines, file_path))

    if language in ("txt", "rst"):
        sections = _chunk_prose_headings(lines, file_path, language)
        return _postprocess_chunks(sections or _fallback_window_chunks(lines, file_path, language))

    if language:
        # UPG-JSFLOW-SYMBOLS: the grammar we PARSE with may differ from `language`
        # (the dict-lookup key below, kept stable so node types resolve the same
        # way) — a Flow-typed .js routes to the tsx grammar.
        parser = _get_parser(_parser_language_for(language, code))
        if parser:
            code_bytes = code.encode("utf-8")
            tree = parser.parse(code_bytes)
            target_types = _CHUNK_NODE_TYPES.get(language, set())
            results: list[CodeChunk] = []
            _collect_chunks_ast(tree.root_node, code_bytes, lines, language, file_path, target_types, results)
            if results:
                return _postprocess_chunks(results)
            # no top-level symbols found → fall through to windows

    # fallback
    lang_label = language or path.suffix.lstrip(".") or "text"
    return _postprocess_chunks(_fallback_window_chunks(lines, file_path, lang_label))
