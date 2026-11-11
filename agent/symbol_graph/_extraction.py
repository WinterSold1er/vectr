"""
Tree-sitter symbol/call-edge extraction and HTTP route extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path

from agent.config import (
    SYMBOL_GRAPH_ERROR_RECOVERY_MAX_EXTEND_STEPS_PER_ATTEMPT,
    SYMBOL_GRAPH_ERROR_RECOVERY_MAX_REPARSE_ATTEMPTS,
    SYMBOL_GRAPH_ERROR_RECOVERY_MIN_SPAN_LINES,
    SYMBOL_GRAPH_RESERVED_KEYWORDS,
)
from agent.symbol_graph._constants import (
    _SYMBOL_TYPES,
    _MODULE_BINDING_TYPES,
    _CALL_TYPES,
    _TYPE_USAGE_NODES,
    _RUST_SKIP_TYPES,
    _MAX_DEPTH,
    _HTTP_METHODS,
    _PY_ROUTE_DECORATOR,
    _PY_ROUTE_METHOD_KW,
    _JAVA_MAPPING,
    _EXPRESS_ROUTE,
)

logger = logging.getLogger(__name__)


def _is_reserved_keyword(name: str, language: str) -> bool:
    """True when `name` is a language keyword for `language` (UPG-JSFLOW-SYMBOLS).

    A desynced/ERROR-node parse (Flow syntax hitting the plain javascript
    grammar, or any other grammar's error-recovery path) can misattribute a
    keyword token — `if`, `for`, `return`, ... — as an identifier. Keyword
    sets are per-language (config.yaml `symbol_graph.reserved_keywords`),
    not a single global list, since keywords differ across language families.
    """
    return name in SYMBOL_GRAPH_RESERVED_KEYWORDS.get(language, frozenset())


# ---------------------------------------------------------------------------
# Tree-sitter helpers (reuse indexer's parser cache)
# ---------------------------------------------------------------------------

def _get_parser(language: str):
    from agent.indexer import _get_parser as _base_get_parser
    return _base_get_parser(language)


def _record_rust_type(name: str) -> bool:
    """A Rust `type_identifier` worth a usage edge: UpperCamelCase (so primitives
    `u32`/`str`/`bool` and snake_case modules are skipped), longer than one char
    (drops generic params `T`/`E`/`K`), and not a std container (UPG-4.4)."""
    return len(name) > 1 and name[0].isupper() and name not in _RUST_SKIP_TYPES


def _rust_call_type_head(func_node, code_bytes: bytes) -> str:
    """Leading type segment of a Rust `Type::assoc(...)` scoped call so
    `trace Type` finds associated-fn and enum-variant construction sites
    (`RegistryClient::new`, `PubGrubPackage::Package`). Returns the rightmost
    path segment for nested paths (`crate::x::RegistryClient::new` → 'RegistryClient')
    or "" when the call isn't a scoped path (UPG-4.4)."""
    if func_node is None or func_node.type != "scoped_identifier":
        return ""
    path = func_node.child_by_field_name("path")
    while path is not None and path.type == "scoped_identifier":
        path = path.child_by_field_name("name")
    if path is not None and path.type in ("identifier", "type_identifier"):
        return code_bytes[path.start_byte:path.end_byte].decode("utf-8", errors="replace")
    return ""


def _get_symbol_name(node, code_bytes: bytes, language: str = "") -> str:
    """Extract identifier from a symbol-defining node."""
    if language in ("c", "cpp"):
        # C/C++ nest the name under the declarator chain; the first direct
        # type_identifier is the RETURN type, not the name — use the shared helper.
        from agent.indexer import c_symbol_name
        return c_symbol_name(node, code_bytes)
    # Go nests a type_declaration's own name one level down, under a type_spec
    # child (`type Point struct {...}` -> type_declaration -> type_spec ->
    # name field) rather than exposing it as a direct child (verified via a
    # live tree-sitter-go parse; UPG-RUST-STRUCT-CHUNK-MISSING). Without this,
    # every Go struct/interface/type-alias symbol silently gets name="" and is
    # dropped from the symbol graph entirely.
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
    # for `impl Trait for Type { }` (`trait` precedes `type`), indexing the
    # impl block under the trait's name instead of the type's. tree-sitter-rust
    # exposes the implemented type as the `type` field for BOTH `impl Type { }`
    # and `impl Trait for Type { }`, so resolve through it, the same field-
    # based approach the qualified Type.method owner lookup already uses
    # (_graph._impl_owner_type_name) and the same shape as the v11 Java fix.
    # A generic impl nests the leaf one level down (`impl<T> Foo<T>` →
    # generic_type whose own `type` field is the bare type_identifier) —
    # unwrap bounded; return "" rather than falling back to the scan, which is
    # exactly the misresolution this branch exists to prevent.
    if language == "rust" and node.type == "impl_item":
        type_node = _rust_impl_type_node(node)
        if type_node is not None:
            return code_bytes[type_node.start_byte:type_node.end_byte].decode("utf-8", errors="replace")
        return ""
    # UPG-JAVA-ENUM-CONSTANTS-NO-L2-SYMBOLS: a Java `field_declaration`
    # has no `name` field — the identifier is nested in the first
    # `variable_declarator`'s `name` field, same shape as the v11 method
    # bug. Without this, the positional child-scan below returns the
    # field's TYPE (the first identifier-ish child in document order)
    # instead of the field's name, and `public int MAX_RETRIES = 3;` is
    # indexed under "int".
    if language == "java" and node.type == "field_declaration":
        name_node = _java_field_name_node(node)
        if name_node is not None:
            return code_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
        return ""
    # Prefer tree-sitter's explicit `name` field when the grammar exposes one.
    # The positional child-scan below returns the FIRST identifier-ish child,
    # which is the RETURN TYPE — not the method name — for a Java
    # `method_declaration` with a non-primitive return type (`Producer foo()`
    # emits `type_identifier` "Producer" before `identifier` "foo"), indexing
    # the method under its return type's name (UPG-JAVA-METHOD-NAME-EXTRACTION).
    # The `name` field is authoritative where present; the scan stays the
    # fallback for node types whose name the grammar doesn't expose that way
    # (e.g. Python `decorated_definition`, Zig `variable_declaration`).
    nm = node.child_by_field_name("name")
    if nm is not None:
        return code_bytes[nm.start_byte:nm.end_byte].decode("utf-8", errors="replace")
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
            return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return ""


_C_SPECIFIER_NAME_NODES = frozenset({
    "struct_specifier", "union_specifier", "enum_specifier", "class_specifier",
    "namespace_definition", "preproc_def", "preproc_function_def",
})


def _c_name_node(node):
    """The name-bearing subtree of a C/C++ symbol node, for scoped error checks
    (UPG-C-STRUCT-TYPEDEF-LOCATE / UPG-C-MACRO-ADJACENT-DROP).

    C nests the identifier under a declarator chain, so the grammar's `name`
    field only exists for the specifier/preproc forms; `function_definition`
    and `type_definition` expose it through `declarator`. Returning the
    declarator (rather than None) lets the caller check the NAME's parse
    integrity in isolation: `typedef struct { SOME_HEADER_MACRO ... } MyStruct;`
    and a macro-body function like `make_value` both error inside the
    body/field list while the declarator (`MyStruct` / `make_value(...)`)
    parses cleanly — the symbol must survive.
    """
    if node.type in _C_SPECIFIER_NAME_NODES:
        return node.child_by_field_name("name")
    return node.child_by_field_name("declarator")


def _rust_impl_type_node(node):
    """The `type_identifier` node naming the implemented type for a Rust
    `impl_item`, or None when the grammar can't resolve one
    (UPG-IMPL-SYMBOL-NAME-NODE-GAP).

    Shared by `_get_symbol_name` (string resolver) and `_get_symbol_name_node`
    (name-node resolver) so the two cannot drift again: before this factor,
    the string resolver walked the `type` field while the name-node resolver
    returned None for `impl_item` and fell back to a whole-subtree error check
    that erased the impl's identity whenever any construct inside the block
    failed to parse. A generic impl nests the leaf one level down
    (`impl<T> Foo<T>` → `generic_type` whose own `type` field is the bare
    `type_identifier`); unwrapped here bounded, so deep or pathological
    nesting returns None rather than guessing.

    `impl Trait for Type { }` always resolves to the type after `for` — the
    `trait` is a separate, optional field present only on that form, so
    resolving through `type` never misreads a trait name.
    """
    if node.type != "impl_item":
        return None
    cur = node.child_by_field_name("type")
    for _ in range(6):  # bounded — generic/reference nesting is shallow
        if cur is None:
            return None
        if cur.type == "type_identifier":
            return cur
        cur = cur.child_by_field_name("type")
    return None


def _get_symbol_name_node(node, language: str):
    """The AST node bearing the symbol's own identifier, when the grammar
    exposes it (UPG-REACT-TSX-FUNCTION-DECL-DROP / UPG-C-STRUCT-TYPEDEF-LOCATE).

    Used to check parse-error status on the NAME ITSELF rather than the whole
    definition subtree: a locally-erroring construct elsewhere in a function's
    signature or body (a Flow-only type the routed grammar can't parse; a
    C macro expanding to non-expression tokens inside a body/struct field list)
    must not erase symbol identity when the name token is clean. For C/C++ the
    name lives in a declarator chain, resolved by `_c_name_node`; for a Rust
    `impl_item` the identifier-bearing subtree is the implemented type's
    `type_identifier` node, resolved by `_rust_impl_type_node` — so the parse-
    error check scopes to the same node the string resolver extracts the name
    from (UPG-IMPL-SYMBOL-NAME-NODE-GAP). Returns None when the node exposes
    no name-bearing subtree — callers then fall back to the broader whole-
    subtree error check.
    """
    if language in ("c", "cpp"):
        return _c_name_node(node)
    if language == "rust" and node.type == "impl_item":
        return _rust_impl_type_node(node)
    if language == "java" and node.type == "field_declaration":
        return _java_field_name_node(node)
    return node.child_by_field_name("name")


def _java_field_name_node(node):
    """The name-bearing subtree of a Java `field_declaration` node, for
    scoped error checks (UPG-JAVA-ENUM-CONSTANTS-NO-L2-SYMBOLS).

    The grammar's `name` field on a `field_declaration` does not exist — the
    field's identifier is nested one level down, in the first
    `variable_declarator`'s own `name` field — so a generic
    `node.child_by_field_name("name")` returns None and the positional
    child-scan fallback would return the FIRST identifier-ish child in
    document order, which is the field's TYPE (`int x = 0;` → `int`, not
    `x`). The same shape as the v11 method-name bug, with the same fix
    shape: walk into the `declarator` field and read its `name`. For
    multi-declarator fields (`int x, y, z;`) the first declarator's name is
    returned — the others aren't separately locatable from a single
    declaration, and the C case has the same property.
    """
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return None
    return declarator.child_by_field_name("name")


# Zig `const X = <container>` RHS node types that make X a genuine type
# definition, mapped to the symbol kind vocabulary (UPG-ZIG-SYMBOL-EXTRACTION).
_ZIG_CONTAINER_KIND: dict[str, str] = {
    "struct_declaration": "struct",
    "union_declaration": "struct",
    "opaque_declaration": "struct",
    "enum_declaration": "enum",
    "error_set_declaration": "enum",
}


# Enclosing-scope kinds whose members belong to a locatable namespace, so a
# value binding nested inside one is a real, addressable symbol (e.g. Zig
# `Node.default_capacity`) — unlike a function body, whose `const`/`var` are
# disposable locals. Mirrors the container kinds `_ZIG_CONTAINER_KIND` emits.
_ZIG_NAMESPACE_KINDS: frozenset[str] = frozenset({"struct", "enum"})


def _zig_var_decl_kind(
    node, code_bytes: bytes, current_symbol: str, name: str, current_kind: str = "",
) -> str | None:
    """Resolve the real symbol kind of a Zig `variable_declaration`, or None to
    skip it (UPG-ZIG-SYMBOL-EXTRACTION).

    Zig's grammar overloads `variable_declaration` across four shapes that the
    static `_SYMBOL_TYPES["zig"]` mapping to "struct" all mislabeled:
      - `const Foo = struct {...}` / `enum {...}` / `union {...}` — a real
        container-type definition → the matching kind.
      - `const std = @import("std")` — an import binding, not a definition here.
      - `const Tree = TreeType(K, V)` / `var checksum: u128 = 0` — a value
        binding; locatable at module scope or as a member of a container
        (struct/enum) namespace, and as a constant/variable, never a
        fabricated "struct".
      - `checksum +%= run();` — a bare assignment the grammar mis-parses as a
        declaration (no `const`/`var` keyword) → not a symbol at all. This was
        the witness: a local mutation indexed as `[struct] checksum` that
        outranked the real `pub fn checksum` definition.

    `current_kind` is the KIND of the enclosing symbol (from the walker), which
    distinguishes a struct/enum namespace member (locatable) from a
    function-local (not) — the two were previously conflated by the
    `current_symbol == ""` gate (UPG-ZIG-STRUCT-CONST-LOCATE).
    """
    child_types = {c.type for c in node.children}
    is_const = "const" in child_types
    if not is_const and "var" not in child_types:
        return None  # bare assignment / mutation, not a declaration
    value = None
    after_eq = False
    for c in node.children:
        if c.type == "=":
            after_eq = True
            continue
        if after_eq and c.is_named:
            value = c
            break
    if value is not None:
        container = _ZIG_CONTAINER_KIND.get(value.type)
        if container is not None:
            return container
        # `@import(...)` binds a module, not a locatable definition.
        if value.type == "builtin_function" and code_bytes[value.start_byte:value.start_byte + 7] == b"@import":
            return None
    # A plain value binding is a locatable symbol at module scope (SCREAMING_CASE
    # heuristic) or as a member of an enclosing container namespace (const vs var
    # taken from the keyword the declaration actually used); a function-local
    # `const`/`var` is not something `locate` looks for.
    if current_symbol == "":
        return "constant" if name.lstrip("_").isupper() else "variable"
    if current_kind in _ZIG_NAMESPACE_KINDS:
        return "constant" if is_const else "variable"
    return None


def _get_call_name(node, code_bytes: bytes) -> str:
    """Extract called function name from a call node."""
    # Python: call → function(identifier | attribute)
    # JS/TS/Go: call_expression → function(identifier | member_expression)
    func = (
        node.child_by_field_name("function")
        or node.child_by_field_name("name")
        or node.child_by_field_name("method")
    )
    if func is None and node.children:
        func = node.children[0]
    if func is None:
        return ""
    if func.type in ("identifier", "property_identifier"):
        return code_bytes[func.start_byte:func.end_byte].decode("utf-8", errors="replace")
    # attribute access: obj.method / ptr->method (C) — extract just the member name
    if func.type in ("attribute", "member_expression", "field_access", "field_expression"):
        fld = func.child_by_field_name("field") or func.child_by_field_name("property")
        if fld is not None:
            return code_bytes[fld.start_byte:fld.end_byte].decode("utf-8", errors="replace")
        for child in func.children:
            if child.type in ("identifier", "property_identifier", "field_identifier") and child != func.children[0]:
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    # fallback: grab last identifier token
    last_ident = ""
    for child in func.children:
        if child.type in ("identifier", "property_identifier"):
            last_ident = code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return last_ident


def _module_binding_names(node, code_bytes: bytes, language: str) -> list[tuple[str, int]]:
    """(name, start_line) for each simple module-level binding target (UPG-10.3).
    Python: the `left` of a top-level `assignment` when it's a bare identifier
    (`X = ...`, `X: T = ...`). Tuple/attribute/subscript targets are skipped —
    those aren't 'definitions' a `locate` is looking for."""
    if language == "python" and node.type == "assignment":
        left = node.child_by_field_name("left")
        if left is not None and left.type == "identifier":
            name = code_bytes[left.start_byte:left.end_byte].decode("utf-8", errors="replace")
            if name:
                return [(name, node.start_point[0] + 1)]
    return []


def _collect_symbols_and_calls(
    node,
    code_bytes: bytes,
    language: str,
    file_path: str,
    symbol_types: dict[str, str],
    call_types: set[str],
    symbols: list[dict],
    edges: list[dict],
    current_symbol: str = "",
    current_line: int = 0,
    current_kind: str = "",
    depth: int = 0,
    type_usage_nodes: set[str] = frozenset(),
    parser=None,
    reparse_budget: list[int] | None = None,
    line_offset: int = 0,
    _just_recovered: bool = False,
    attempted_spans: set[tuple[int, int]] | None = None,
) -> None:
    """Recursively walk AST collecting symbols and call edges.

    `parser`/`reparse_budget`/`line_offset` support UPG-REACT-TSX-FUNCTION-DECL-DROP
    error-recovery reparsing (see the recovery branch below); `code_bytes` is
    whatever byte range the CURRENT frame is walking (the whole file at the
    top level, or an isolated reparsed sub-blob when recovering), and
    `line_offset` is the cumulative row offset from that frame back to the
    true file so emitted `start_line`/`end_line` stay correct. `attempted_spans`
    memoizes (absolute start line, absolute end line) pairs already sent
    through a recovery reparse: a fragment that is genuinely incomplete on
    its own (e.g. a mid-expression excerpt with no enclosing statement)
    reparses into an identically-shaped, identically-errored wrapper node
    covering the same lines — without this guard that reproduces itself
    every recursion and burns the whole budget on one useless span.
    """
    if depth > _MAX_DEPTH:
        return
    if reparse_budget is None:
        reparse_budget = [SYMBOL_GRAPH_ERROR_RECOVERY_MAX_REPARSE_ATTEMPTS]
    if attempted_spans is None:
        attempted_spans = set()
    if node.type in symbol_types:
        name = _get_symbol_name(node, code_bytes, language)
        kind = symbol_types[node.type]
        start = node.start_point[0] + 1 + line_offset
        end = node.end_point[0] + 1 + line_offset
        # UPG-ZIG-SYMBOL-EXTRACTION: Zig's `variable_declaration` covers real
        # container-type defs, value bindings, imports, and mis-parsed
        # assignments — resolve the true kind (None → not a locatable symbol).
        if language == "zig" and node.type == "variable_declaration":
            kind = _zig_var_decl_kind(node, code_bytes, current_symbol, name, current_kind)
        # UPG-JSFLOW-SYMBOLS / UPG-REACT-TSX-FUNCTION-DECL-DROP: skip anonymous
        # nodes (e.g. C anonymous struct inside a typedef), language keywords
        # misattributed as identifiers, and any node whose own NAME token comes
        # from a parse error — a corrupted/desynced parse (e.g. Flow syntax the
        # grammar can't parse) must not mint a symbol from a bogus identifier.
        # A parse error CONTAINED elsewhere in the subtree (a locally-unparseable
        # type in a parameter or the body) does not erase a legitimate
        # declaration whose own name is clean — checked on the name node in
        # isolation when the grammar exposes one; the previous, broader
        # whole-subtree check remains the fallback where it doesn't.
        name_node = _get_symbol_name_node(node, language)
        name_is_junk = name_node.has_error if name_node is not None else node.has_error
        if name and kind is not None and not _is_reserved_keyword(name, language) and not name_is_junk:
            symbols.append({
                "name": name,
                "kind": kind,
                "file_path": file_path,
                "start_line": start,
                "end_line": end,
            })
        # recurse into body with this symbol as context. Only a symbol we
        # actually emitted names the context; a skipped Zig local/mutation
        # (kind None) must not capture the calls in its initializer — those
        # belong to the enclosing function. For non-Zig nodes kind is never
        # None, so this is byte-identical to the prior `name or current_symbol`.
        emitted = bool(name) and kind is not None
        ctx = name if emitted else current_symbol
        ctx_line = start if emitted else current_line
        ctx_kind = kind if emitted else current_kind
        for child in node.children:
            _collect_symbols_and_calls(
                child, code_bytes, language, file_path,
                symbol_types, call_types, symbols, edges,
                current_symbol=ctx, current_line=ctx_line, current_kind=ctx_kind,
                depth=depth + 1, type_usage_nodes=type_usage_nodes,
                parser=parser, reparse_budget=reparse_budget, line_offset=line_offset,
                attempted_spans=attempted_spans,
            )
        return

    # UPG-REACT-TSX-FUNCTION-DECL-DROP: a single unparseable construct can
    # desync a grammar's error recovery badly enough that an unrelated span of
    # SIBLING declarations gets swallowed into one opaque, mis-typed node
    # (e.g. tree-sitter emits a bogus `member_expression`/`ERROR` covering
    # hundreds of lines instead of the real `function_declaration` nodes
    # within it) — not just a locally-contained error on one declaration.
    # Reparsing that node's own byte range in isolation frequently resyncs
    # cleanly: the parser starts fresh, no longer carrying the earlier
    # desync's state. Language-agnostic — gated purely on node shape (opaque
    # + errored + large), not on any language/keyword content. A fragment
    # that ISN'T a complete top-level unit on its own (e.g. a mid-expression
    # excerpt) reparses right back into an identically-shaped, identically-
    # errored wrapper covering the same lines — `attempted_spans` recognizes
    # that no-progress case and gives up on that span after one try instead
    # of re-triggering on the lookalike wrapper every recursion.
    span_key = (node.start_point[0] + line_offset, node.end_point[0] + line_offset)
    if (
        parser is not None
        and not _just_recovered
        and node.has_error
        and reparse_budget[0] > 0
        and span_key not in attempted_spans
        and (node.end_point[0] - node.start_point[0]) >= SYMBOL_GRAPH_ERROR_RECOVERY_MIN_SPAN_LINES
    ):
        reparse_budget[0] -= 1
        attempted_spans.add(span_key)
        end_byte = node.end_byte
        sibling = node.next_sibling
        blob = code_bytes[node.start_byte:end_byte]
        sub_tree = parser.parse(blob)
        # The desync's error-recovery boundary for `node` is an arbitrary
        # token cut, not a real statement boundary — it can land mid-
        # declaration, leaving the isolated reparse's own trailing child
        # errored too. Grow the reparsed range to absorb the next ORIGINAL
        # sibling (tree-sitter's own sibling link — a structural move, no
        # content/keyword matching) and retry until the tail clears or the
        # shared budget runs out, so a declaration split across the cut is
        # recovered whole.
        # Extension steps are bounded by their own small per-attempt cap, not
        # the shared per-file budget — otherwise one badly-cut region could
        # spend the entire file's budget on itself and starve every other
        # desynced region.
        extend_steps = 0
        while (
            sub_tree.root_node.children
            and sub_tree.root_node.children[-1].has_error
            and sibling is not None
            and extend_steps < SYMBOL_GRAPH_ERROR_RECOVERY_MAX_EXTEND_STEPS_PER_ATTEMPT
        ):
            extend_steps += 1
            end_byte = sibling.end_byte
            sibling = sibling.next_sibling
            blob = code_bytes[node.start_byte:end_byte]
            sub_tree = parser.parse(blob)
        _collect_symbols_and_calls(
            sub_tree.root_node, blob, language, file_path,
            symbol_types, call_types, symbols, edges,
            current_symbol=current_symbol, current_line=current_line,
            current_kind=current_kind,
            depth=depth + 1, type_usage_nodes=type_usage_nodes,
            parser=parser, reparse_budget=reparse_budget,
            line_offset=line_offset + node.start_point[0],
            _just_recovered=True,
            attempted_spans=attempted_spans,
        )
        return

    # UPG-10.3: module-level constant/variable bindings. ONLY at module scope —
    # `current_symbol == ""` means we're not inside any function or class (those
    # set the context), so locals can never leak in. Don't return: the value side
    # may still contain calls/symbols worth walking.
    if not current_symbol and node.type in _MODULE_BINDING_TYPES.get(language, frozenset()):
        for nm, ln in _module_binding_names(node, code_bytes, language):
            symbols.append({
                "name": nm,
                "kind": "constant" if nm.lstrip("_").isupper() else "variable",
                "file_path": file_path,
                "start_line": ln + line_offset,
                "end_line": node.end_point[0] + 1 + line_offset,
            })

    if node.type in call_types and current_symbol:
        callee = _get_call_name(node, code_bytes)
        # UPG-JSFLOW-SYMBOLS: same reserved-keyword guard as symbol emission —
        # a keyword token misattributed as a call name (desynced/ERROR-node
        # parse) must not mint a call edge either.
        if callee and not _is_reserved_keyword(callee, language):
            edges.append({
                "from_file": file_path,
                "from_symbol": current_symbol,
                "from_line": current_line,
                "to_symbol": callee,
                "edge_type": "calls",
            })
        # UPG-4.4: `Type::assoc(...)` also links the caller to the TYPE so
        # `trace Type` finds construction / enum-variant sites — not just the
        # bare `new`/`Package` method name `_get_call_name` returns above.
        if type_usage_nodes:
            head = _rust_call_type_head(node.child_by_field_name("function"), code_bytes)
            if head and head != current_symbol and _record_rust_type(head):
                edges.append({
                    "from_file": file_path,
                    "from_symbol": current_symbol,
                    "from_line": current_line,
                    "to_symbol": head,
                    "edge_type": "uses",
                })

    # UPG-4.4: a type reference in a signature/field/generic position links the
    # enclosing symbol to that type. Don't return — generic args nest further
    # type_identifiers (`Result<RegistryClient, Error>`) reached by recursion.
    if node.type in type_usage_nodes and current_symbol:
        tname = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if tname != current_symbol and _record_rust_type(tname):
            edges.append({
                "from_file": file_path,
                "from_symbol": current_symbol,
                "from_line": current_line,
                "to_symbol": tname,
                "edge_type": "uses",
            })

    for child in node.children:
        _collect_symbols_and_calls(
            child, code_bytes, language, file_path,
            symbol_types, call_types, symbols, edges,
            current_symbol=current_symbol, current_line=current_line,
            current_kind=current_kind,
            depth=depth + 1,  # MUST increment — generic nodes dominate deep C ASTs;
                              # leaving this at `depth` let the guard never fire → RecursionError
            type_usage_nodes=type_usage_nodes,
            parser=parser, reparse_budget=reparse_budget, line_offset=line_offset,
            attempted_spans=attempted_spans,
        )


def extract_symbols_from_file(file_path: str) -> tuple[list[dict], list[dict]]:
    """
    Parse a source file and return (symbols, edges).
    Symbols: list of {name, kind, file_path, start_line, end_line}
    Edges:   list of {from_file, from_symbol, from_line, to_symbol, edge_type}
    """
    from agent.indexer import LANG_BY_EXT, _parser_language_for
    path = Path(file_path)
    language = LANG_BY_EXT.get(path.suffix.lower(), "")
    if not language:
        return [], []

    try:
        code = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], []

    # UPG-JSFLOW-SYMBOLS: the grammar we PARSE with may differ from `language`
    # (the dict-lookup key used below for _SYMBOL_TYPES/_CALL_TYPES, kept
    # stable so the desugared node types resolve the same way) — a
    # Flow-typed .js routes to the tsx grammar, which parses type
    # annotations instead of desyncing into ERROR nodes.
    parser = _get_parser(_parser_language_for(language, code))
    if parser is None:
        return [], []

    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)

    symbol_types = _SYMBOL_TYPES.get(language, {})
    call_types = _CALL_TYPES.get(language, set())
    type_usage_nodes = _TYPE_USAGE_NODES.get(language, frozenset())

    symbols: list[dict] = []
    edges: list[dict] = []
    _collect_symbols_and_calls(
        tree.root_node, code_bytes, language, file_path,
        symbol_types, call_types, symbols, edges,
        type_usage_nodes=type_usage_nodes,
        parser=parser,
    )

    # deduplicate symbols — UPG-REACT-TSX-FUNCTION-DECL-DROP's error-recovery
    # reparse can grow its byte range across an original sibling boundary to
    # recover a declaration split by an arbitrary parser-recovery cut; that
    # sibling's content may then also be reached a second time via the
    # normal walk of the (still error-free-looking) original tree, so the
    # same symbol can be emitted twice.
    seen_symbols = set()
    deduped_symbols: list[dict] = []
    for s in symbols:
        key = (s["file_path"], s["name"], s["kind"], s["start_line"], s["end_line"])
        if key not in seen_symbols:
            seen_symbols.add(key)
            deduped_symbols.append(s)
    symbols = deduped_symbols

    # deduplicate edges
    seen = set()
    deduped_edges: list[dict] = []
    for e in edges:
        key = (e["from_file"], e["from_symbol"], e["to_symbol"])
        if key not in seen:
            seen.add(key)
            deduped_edges.append(e)

    # UPG-BASE-METHOD-OVERRIDE-FLOOD: emit edges of edge_type="overrides" from a
    # subclass method to a same-named method on a base class defined in the
    # same file. Recorded as facts about the code (structural analysis), NOT
    # query-side heuristics. Currently scoped to same-file resolution only —
    # see LANE-REPORT.md for the precision/recall tradeoff and the rejection
    # of cross-file inference here.
    override_edges = _extract_overrides(
        tree.root_node, code_bytes, language, file_path,
    )
    # Dedup override edges by the FULL key (file, source symbol, source line,
    # target symbol, edge type) — not the call-edge dedup key above, which
    # drops `from_line` and would collapse two distinct same-named subclass
    # methods (e.g. two `verify` overrides at different lines) into one.
    # The `edges` table's own unique index would dedup a duplicate INSERT
    # silently anyway, but doing it in Python keeps the return value
    # honest for tests that count returned edges without touching the DB.
    seen_overrides: set = set()
    for e in override_edges:
        ok = (
            e["from_file"], e["from_symbol"],
            e["from_line"], e["to_symbol"], e["edge_type"],
        )
        if ok not in seen_overrides:
            seen_overrides.add(ok)
            deduped_edges.append(e)

    # extract HTTP route symbols (Flask/FastAPI/Express/Spring)
    symbols.extend(_extract_routes(file_path, code, language))

    return symbols, deduped_edges


# ---------------------------------------------------------------------------
# HTTP route extraction — framework-aware route nodes
#
# Extracts route symbols from common web frameworks and adds them to the L2
# symbol graph with kind="route". This makes routes navigable via vectr_locate
# and searchable without reading controller/view files.
#
# Supported frameworks:
#   Python: Flask (@app.route, @app.get/post/...), FastAPI (@router.get/post/...)
#   Java:   Spring @GetMapping, @PostMapping, @PutMapping, @DeleteMapping, @RequestMapping
#   JS/TS:  Express (app.get/post/..., router.get/post/...)
# ---------------------------------------------------------------------------

def _extract_routes(file_path: str, source: str, language: str) -> list[dict]:
    """Return a list of route symbol dicts extracted from source code."""
    routes: list[dict] = []
    lines = source.splitlines()

    if language == "python":
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _PY_ROUTE_DECORATOR.search(line)
            if m:
                verb_from_decorator = m.group(1).upper()
                path = m.group(2)

                # If it's @app.route(...), also look for methods=[] kwarg on same line
                methods: list[str] = []
                if verb_from_decorator == "ROUTE":
                    kw = _PY_ROUTE_METHOD_KW.search(line)
                    if kw:
                        raw_methods = kw.group(1)
                        methods = [v.strip().strip("\"'").upper() for v in raw_methods.split(",")]
                    else:
                        methods = ["GET"]  # Flask default
                else:
                    methods = [verb_from_decorator]

                for method in methods:
                    routes.append({
                        "name": f"{method} {path}",
                        "kind": "route",
                        "file_path": file_path,
                        "start_line": i + 1,
                        "end_line": i + 1,
                    })
            i += 1

    elif language == "java":
        for i, line in enumerate(lines):
            m = _JAVA_MAPPING.search(line)
            if m:
                annotation = m.group(1).upper()
                path = m.group(2)
                method = "GET" if annotation == "REQUEST" else annotation
                routes.append({
                    "name": f"{method} {path}",
                    "kind": "route",
                    "file_path": file_path,
                    "start_line": i + 1,
                    "end_line": i + 1,
                })

    elif language in ("javascript", "typescript"):
        for i, line in enumerate(lines):
            m = _EXPRESS_ROUTE.search(line)
            if m:
                method = m.group(1).upper()
                path = m.group(2)
                routes.append({
                    "name": f"{method} {path}",
                    "kind": "route",
                    "file_path": file_path,
                    "start_line": i + 1,
                    "end_line": i + 1,
                })

    return routes


# ---------------------------------------------------------------------------
# UPG-BASE-METHOD-OVERRIDE-FLOOD: same-file class-base override edges.
#
# Records edges of edge_type="overrides" from a method on a subclass to the
# same-named method on a base class defined in the SAME file. The data this
# writes into the existing `edges` table is the structural input a later
# ranking change would need to break a same-leaf top-k tie in favour of the
# base method (the F23 / F56 / F1 family of "overrides crowd out the
# canonical base" cases). This lane ships the data; it deliberately stops
# short of changing any ranking/scoring/pool-composition behaviour, since
# that needs measurement, and the measurement needs the data.
#
# Resolution scope and why it is narrow (full argument in LANE-REPORT.md):
#
#   1. A class base named in the same file as the subclass is resolvable —
#      we already have all the symbols in the file from this same extraction
#      pass, so resolving "Sub.method → Base.method" is a local lookup.
#
#   2. A class base imported from elsewhere in the workspace COULD be
#      resolved by walking the existing import machinery (_get_imported_files
#      in _graph.py) and re-extracting the imported file. It is not, here.
#      A wrong override edge would later push the wrong symbol up — a worse
#      outcome than a missing one. Cross-file base resolution needs a real
#      type resolver (or at minimum, a per-file import-aware lookup with
#      cycle-safety and cross-file symbol-id mapping) that is not in scope
#      for this lane; the F56/F23 witnesses are all single-file (Django
#      has the canonical base class definition in the same file as the
#      overrides), so same-file resolution captures the immediate target
#      while leaving the bigger problem for a later lane.
#
#   3. A class base from a third-party package, or a dynamically constructed
#      one, is not resolvable here — record nothing rather than guess.
#
# Implemented languages: Python and JavaScript/TypeScript (both expose a
# single `class ... (Base)` form with a list of base-class identifiers; both
# are in the Django fixture for the F56 case via Python, and TS class
# heritage is a common ask). Java/C++/Rust/Go use a different syntax shape
# (interface lists, multi-section `class Foo : public Bar, virtual Baz`,
# trait/impl blocks) and are out of scope for this lane.
# ---------------------------------------------------------------------------


def _python_class_bases(node, code_bytes: bytes) -> list[str]:
    """Extract the simple-identifier base class names from a Python
    `class_definition` node's `superclasses` field.

    Only plain identifier bases are returned. Compound bases like
    `Foo.Bar` (attribute lookup, common in `class X(models.Model)`) and
    `Foo[T]` (generic) are skipped — without a real name resolver the
    identifier on the left of a `.` or inside `[...]` does not name a
    single class. The remaining class names are the ones the same-file
    override lookup below can actually try to resolve.
    """
    bases: list[str] = []
    sup = node.child_by_field_name("superclasses")
    if sup is None:
        return bases
    for child in sup.children:
        # tree-sitter-python: each base is a direct child of the
        # superclasses node — an `identifier` for plain bases, an
        # `attribute` for `pkg.Class`, a `subscript` for `Generic[T]`.
        if child.type == "identifier":
            bases.append(
                code_bytes[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                )
            )
    return bases


def _js_class_bases(node, code_bytes: bytes) -> list[str]:
    """Extract the simple-identifier base class names from a JS/TS
    `class_declaration` node's `class_heritage` field.

    Same shape as the Python case: the `class_heritage` node wraps a list
    of base-class expressions. Only plain `identifier` bases (e.g.
    `class Foo extends Bar` or `class X extends Bar, Baz`) are returned.
    `extends` keyword nodes and `class_heritage` itself are skipped.
    """
    bases: list[str] = []

    # `class_heritage` is a child NODE TYPE, not a named field, so
    # child_by_field_name("class_heritage") returns None on both grammars
    # (class_declaration exposes only `name` and `body`). Scan children.
    heritage = None
    for child in node.children:
        if child.type == "class_heritage":
            heritage = child
            break
    if heritage is None:
        return bases

    def _text(n) -> str:
        return code_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    for child in heritage.children:
        # javascript puts the base identifier directly under class_heritage
        # (`extends` keyword, then `identifier`).
        if child.type == "identifier":
            bases.append(_text(child))
        # typescript wraps it: class_heritage -> extends_clause -> identifier.
        elif child.type == "extends_clause":
            for grand in child.children:
                if grand.type == "identifier":
                    bases.append(_text(grand))
        # `implements_clause` names INTERFACES, not a superclass. A method
        # declared on an interface has no body to override, so an edge to it
        # would not point at a base definition. Deliberately skipped.
    return bases


# Public-facing list of which languages support the same-file override
# extraction in this lane. The map is intentionally a small two-entry
# dispatch table, not a generic per-language registry — adding a new
# language here means the same commit has to define the base-list
# extractor AND verify it on real code, both of which deserve explicit
# reviewer eyes. Other languages get a no-op (return []).
_BASE_EXTRACTORS = {
    "python": _python_class_bases,
    "javascript": _js_class_bases,
    "typescript": _js_class_bases,
}


def _extract_overrides(
    root_node,
    code_bytes: bytes,
    language: str,
    file_path: str,
) -> list[dict]:
    """Return override edges for class methods in this file.

    A class base is resolvable only when the base is defined in the same
    file (a same-file symbol). For each top-level class in *file_path* that
    has resolvable bases, every method defined directly inside the class
    that has a same-named method on at least one base produces an
    `overrides` edge from the subclass method to the base method.

    Returns [] for languages that don't expose a `class ... (Base)` form
    (Rust/Go use impl blocks / method-declaration-on-receiver types; C++
    uses a multi-section base-clause; Java uses `extends`/`implements` —
    all deliberately skipped here).
    """
    base_extractor = _BASE_EXTRACTORS.get(language)
    if base_extractor is None:
        return []

    # Method index: (class_name) -> {method_name: start_line}, built by
    # walking the AST (NOT from the symbols table) so we know the
    # IMMEDIATE parent class of each method without re-deriving it from
    # the body's own structure. The symbol dicts that
    # `_collect_symbols_and_calls` emits do not carry an `enclosing_class`
    # field — class attribution is implicit in the walk context — so the
    # table lookup that would normally answer "what class owns this
    # method" needs a fresh walk to determine the parent chain at this
    # granularity (immediate parent vs transitive ancestor matters for
    # an inner-class-override-inner-class case).
    methods_by_class: dict[str, dict[str, int]] = {}
    _collect_methods_by_class(root_node, code_bytes, language, methods_by_class)

    edges: list[dict] = []
    stack = [root_node]
    while stack:
        node = stack.pop()
        for child in node.children:
            stack.append(child)
        if node.type in ("class_definition", "class_declaration"):
            bases = base_extractor(node, code_bytes)
            if not bases:
                continue
            subclass_name = _get_symbol_name(node, code_bytes, language)
            if not subclass_name:
                continue
            # The own_methods for this class are already known from the
            # _collect_methods_by_class pass — look them up rather than
            # re-walking the body. _collect_subclass_methods is kept for
            # future callers that need the (name, line) tuple form
            # directly, but the override-edge path uses the index.
            own_methods_dict = methods_by_class.get(subclass_name, {})
            own_methods = list(own_methods_dict.items())
            # For each base present in this file, find same-named methods
            # on the base and emit an overrides edge per (subclass method,
            # base method) pair. A class can extend multiple bases (Python
            # MRO; TS `extends A, B, C`); the "this is the canonical base"
            # question is for a later ranking change to answer — recording
            # every candidate override is the data-only step this lane
            # ships.
            for base_name in bases:
                base_methods = methods_by_class.get(base_name)
                if not base_methods:
                    continue
                for mname, mline in own_methods:
                    base_line = base_methods.get(mname)
                    if base_line is None:
                        continue
                    # Edge naming convention: the source `from_symbol` is
                    # the bare method name to match what the symbols table
                    # stores and what the existing `calls`/`uses` edge
                    # convention uses — consumers identify the SOURCE
                    # method by (from_file, from_symbol, from_line), which
                    # is exact even when many classes in the same file
                    # share the same method name. The TARGET is qualified
                    # `BaseClass.method` so the consumer can locate it
                    # unambiguously (the F56 case has e.g. six subclasses
                    # each overriding `verify`; a bare `verify -> verify`
                    # edge would lose which class owns the target).
                    # The base lives in the same file as the source
                    # (same-file scope is the explicit precision/recall
                    # tradeoff of this lane — see LANE-REPORT); the
                    # consumer can resolve the base line via the `symbols`
                    # table on (file_path, name="BaseClass.method") or
                    # via `locate("BaseClass.method")`.
                    edges.append({
                        "from_file": file_path,
                        "from_symbol": mname,
                        "from_line": mline,
                        "to_symbol": f"{base_name}.{mname}",
                        "edge_type": "overrides",
                    })
    return edges


def _collect_methods_by_class(
    node,
    code_bytes: bytes,
    language: str,
    out: dict[str, dict[str, int]],
) -> None:
    """Walk the AST and populate *out* with `ClassName -> {method_name: start_line}`
    for every class-shaped node in the file. The class name is the immediate
    parent's identity — a method defined in an inner class is attributed to
    the inner class, not its outer class.

    Limitation: a class definition is given a SIMPLE name (its own identifier),
    not a qualified path. Two nested classes with the same simple name
    (e.g. an inner `class Meta` in two different outer classes) would
    collide on the simple name. This is consistent with how the rest of
    vectr handles class names (the locate surface has the same
    simplification) and is a deliberate scope decision — the F56/F23
    cases do not exercise nested-class-name collisions, and a qualified
    name would force a parallel naming convention just for the override
    edge target.
    """
    if node.type in ("class_definition", "class_declaration"):
        cls_name = _get_symbol_name(node, code_bytes, language)
        if cls_name and cls_name not in out:
            out[cls_name] = {}
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                if child.type in (
                    "function_definition", "method_definition",
                    "function_declaration", "method_declaration",
                ):
                    if cls_name:
                        mname = _get_symbol_name(child, code_bytes, language)
                        if mname and mname not in out[cls_name]:
                            out[cls_name][mname] = (
                                child.start_point[0] + 1
                            )
                # Recurse into inner classes — they are also resolvable
                # bases (a subclass can extend an inner-class type).
                if child.type in ("class_definition", "class_declaration"):
                    _collect_methods_by_class(child, code_bytes, language, out)
        return  # do not double-recurse; the body is fully walked above
    for child in node.children:
        _collect_methods_by_class(child, code_bytes, language, out)

