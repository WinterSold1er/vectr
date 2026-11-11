"""
Module-level constants, language maps, regex patterns, and metadata functions
for the SymbolGraph package.
"""
from __future__ import annotations

import re

from agent.config import OUTPUT_SNIPPET_LINES as SNIPPET_LINES


# Node types that define symbols per language
_SYMBOL_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "function",
    },
    "javascript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "arrow_function": "function",
    },
    "typescript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        # type_alias_declaration/enum_declaration are standalone TS type
        # definitions the L3 chunker already collects (_CHUNK_NODE_TYPES); the
        # L2 symbol graph must resolve them too so `vectr_locate` finds a TS
        # type alias or enum, not just interfaces/classes (UPG-TS-SYMBOLGRAPH-TYPEDEF).
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "arrow_function": "function",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "struct",
    },
    "rust": {
        "function_item": "function",
        # An impl block is an implementation, not the type's definition. Keep it a
        # distinct kind so locate can rank the `struct`/`enum`/`trait` def ahead of
        # the (often many) impl blocks that share the type's name (UPG-4.5).
        "impl_item": "impl",
        "struct_item": "struct",
        "trait_item": "interface",
        "enum_item": "enum",
    },
    "java": {
        "method_declaration": "method",
        # UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE: a constructor is a member
        # function like any other method — kind "method" keeps it consistent
        # with method_declaration, and _KIND_RANK still ranks the class itself
        # (0) above its constructor (1) when both share the class's name.
        "constructor_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        # UPG-JAVA-ENUM-CONSTANTS-NO-L2-SYMBOLS: enum constants and class
        # fields are members of a type's namespace, parallel to methods —
        # tree-sitter-java separates `field_declaration` (class/enum body
        # members) from `local_variable_declaration` (method body locals) at
        # the grammar level, so the enclosing-kind gate the Zig case
        # required isn't needed here: locals never appear as a
        # `field_declaration` node, so adding this entry cannot leak
        # method-locals into the graph. Kinds rank at 1 alongside "method"
        # so the enum/class that contains them (rank 0) is still the
        # canonical answer to "where is X defined" when both share a name.
        "enum_constant": "enum_constant",
        "field_declaration": "field",
    },
    "zig": {
        "function_declaration": "function",
        "variable_declaration": "struct",  # pub const Foo = struct { ... }
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "union_specifier": "struct",
        "enum_specifier": "enum",
        "type_definition": "type",       # typedef … Name;
        "preproc_def": "macro",
        "preproc_function_def": "macro",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "union_specifier": "struct",
        "enum_specifier": "enum",
        "type_definition": "type",
        "namespace_definition": "namespace",
        "preproc_def": "macro",
        "preproc_function_def": "macro",
    },
}

# Symbol `kind` values (the presentational labels _SYMBOL_TYPES maps node types
# to, above) that name-frequency importance is computed over
# (SymbolGraph._compute_and_store_class_importance, ARCH-2 /
# UPG-SIBLING-TYPEDEF-CROWDING). Covers every TYPE-DEFINITION kind across
# languages (class, struct, enum, interface — Rust traits map to "interface" —
# and C/C++'s "type" for typedef/using) plus "function" for module-level
# functions. Deliberately excludes "method" (always attributed via its owning
# class, not its own name — see searcher._apply_quality_and_dedup), "impl"
# (an implementation of a type, not the type's own definition — UPG-4.5), and
# "route"/"macro"/"namespace" (not the corpus-centrality signal this table
# targets). Derived directly from the kind vocabulary above, not a tunable a
# reviewer would adjust independent of the parser/grammar it describes.
_IMPORTANCE_SYMBOL_KINDS: frozenset[str] = frozenset({
    "class", "struct", "enum", "interface", "type", "function",
})

# Node types per language whose `_SYMBOL_TYPES` kind is "impl" — a container
# that IMPLEMENTS a type (Rust's `impl_item`) rather than defining one, so a
# member declared inside it is owned by the implemented type, not by the
# container node's own identity. Derived directly from `_SYMBOL_TYPES` (not a
# separate hardcoded per-language check) so any future "impl"-kind language is
# picked up automatically by the enclosing-owner lookup in
# `agent/symbol_graph/_graph.py` (UPG-RUST-IMPL-QUALIFIED-LOCATE) without
# touching that lookup's own code.
_IMPL_CONTAINER_NODE_TYPES: dict[str, frozenset[str]] = {
    lang: frozenset(nt for nt, kind in mapping.items() if kind == "impl")
    for lang, mapping in _SYMBOL_TYPES.items()
}

# UPG-10.3: node types that bind a MODULE-LEVEL name (a constant/config/binding
# that isn't a function or class but IS something callers `locate` — e.g. Python
# `_CLAUDE_MD = """..."""`). Indexed only at module scope (see the scope guard in
# _collect_symbols_and_calls) so function locals never flood the graph. Python
# only for now; JS/TS/Rust/Go top-level const/static are a follow-up (Zig top-
# level `pub const Foo = ...` is already covered via _SYMBOL_TYPES["zig"]).
_MODULE_BINDING_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"assignment"}),
}

# Languages vectr can extract a symbol graph for — i.e. where locate/trace work.
# Anything outside this set is search-only (chunks are indexed, but there are no
# symbol/call-graph edges). UPG-3.3 surfaces this per-language so the caller LLM
# can route: use locate/trace where symbols exist, fall back to search elsewhere.
SYMBOL_LANGUAGES: frozenset[str] = frozenset(_SYMBOL_TYPES)

# Intentionally NOT in config.yaml (Tier-3): SYMBOL_SCHEMA_VERSION is a
# schema-migration trigger.  Changing it via config would silently corrupt or
# force a full reindex without the usual version-bump safeguards.
# Bump whenever symbol/edge extraction changes in a way that makes an
# already-persisted graph stale (new parser language, new edge type, changed
# name resolution). Combined with the parser-language set + embed model into the
# toolchain fingerprint (UPG-8.7) so a vectr upgrade is detectable and the graph
# is rebuilt rather than silently serving partial/old results.
SYMBOL_SCHEMA_VERSION = 15  # 1: base · 2: C/C++ + per-def trace (UPG-3.2/4.x) · 3: Rust uses-edges (UPG-4.4) · 4: module-level constants (UPG-10.3) · 5: .txt/.rst prose docs indexed (UPG-11.3) · 6: symbol_importance table added (ARCH-1a) · 7: Flow-typed .js routed to tsx grammar + keyword/ERROR-node symbol rejection (UPG-JSFLOW-SYMBOLS) · 8: class_importance table added (ARCH-2) · 9: name-node-scoped error check (a locally-erroring construct no longer erases a symbol whose own name token is clean) + isolated-reparse error-recovery for catastrophically desynced subtrees (UPG-REACT-TSX-FUNCTION-DECL-DROP) · 10: class_importance seeded from all type-definition kinds (struct/enum/interface/type, not just class) plus module-level functions, and normalized with log(1+count)/log(1+max) instead of linear (UPG-SIBLING-TYPEDEF-CROWDING) · 11: Java method name via `name` field not return-type child-scan (UPG-JAVA-METHOD-NAME-EXTRACTION); C/C++ declarator-scoped error check so macro-body functions/typedef structs survive (UPG-C-STRUCT-TYPEDEF-LOCATE/UPG-C-MACRO-ADJACENT-DROP); Zig variable_declaration kind resolved by RHS, locals/mutations/imports no longer indexed as [struct] (UPG-ZIG-SYMBOL-EXTRACTION); TS type_alias_declaration/enum_declaration added to symbol graph (UPG-TS-SYMBOLGRAPH-TYPEDEF) · 12: Zig struct/enum-scoped const/var extracted as locatable namespace members (function-locals still excluded) — enclosing symbol kind threaded through the walker (UPG-ZIG-STRUCT-CONST-LOCATE) · 13: Rust impl_item name via grammar `type` field not positional child-scan — `impl Trait for Type { }` indexed under the TYPE, never the trait (UPG-RUST-IMPL-SYMBOL-NAME-TRAIT-VS-TYPE); Java constructor_declaration extracted as a method-kind symbol so constructors are locatable and call-attributable like any member function (UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE) · 14: Java enum_constant and field_declaration extracted as symbols so `locate` finds enum values and class fields the L3 search already covers — tree-sitter-java separates `field_declaration` (class/enum body members) from `local_variable_declaration` (method body locals) at the grammar level, so no enclosing-kind gate is needed (UPG-JAVA-ENUM-CONSTANTS-NO-L2-SYMBOLS); Rust impl_item parse-error check now scoped to the implemented type's `type_identifier` node (one shared walk with the string resolver, not the whole impl block) so an impl containing a locally-erroring construct keeps its symbol identity the way an equivalent function does (UPG-IMPL-SYMBOL-NAME-NODE-GAP) · 15: same-file class-base override edges of edge_type="overrides" recorded at index time; SymbolGraph.overrides_of / SymbolGraph.overridden_by expose the read API (UPG-BASE-METHOD-OVERRIDE-FLOOD). Same-file scope only — see LANE-REPORT for the precision/recall tradeoff and the explicit rejection of cross-file inference here


def grammar_available(language: str) -> bool:
    """True iff the tree-sitter grammar for `language` actually loads in this environment.

    Probes by calling agent.indexer._get_parser(language) — function-level import
    to avoid import cycles. Normalises display names the same way supports_symbols
    does so either form ("C++", "cpp") works.

    A False result means the grammar package is declared in pyproject.toml but not
    installed (e.g. an editable install predating a grammar being added). In that
    case locate/trace must be advertised as unavailable for that language and the
    toolchain fingerprint must exclude it so installing the grammar later triggers
    a full graph rebuild.
    """
    if not language:
        return False
    norm = language.strip().lower()
    norm = {"c++": "cpp", "cplusplus": "cpp", "objective-c": "c"}.get(norm, norm)
    if norm not in SYMBOL_LANGUAGES:
        return False
    # Function-level import to avoid circular dependency (agent.indexer imports
    # from agent.symbol_graph via _extraction; we access it at call-time only).
    from agent.indexer import _get_parser  # noqa: PLC0415
    return _get_parser(norm) is not None


def available_symbol_languages() -> frozenset:
    """Subset of SYMBOL_LANGUAGES whose tree-sitter grammar successfully loads.

    Returns a frozenset[str] of normalised language keys. Languages declared in
    SYMBOL_LANGUAGES but whose grammar package is missing from the environment
    are excluded. Used by graph_toolchain_fingerprint so that installing or
    removing a grammar changes the fingerprint and triggers a graph rebuild.
    """
    return frozenset(lang for lang in SYMBOL_LANGUAGES if grammar_available(lang))


def graph_toolchain_fingerprint(embed_model: str = "") -> str:
    """Identity of the toolchain that builds the symbol graph.

    A change here means an already-persisted graph was built by a different
    vectr (new/changed parser, bumped schema, different embed model) and must be
    rebuilt — otherwise locate/trace silently serve stale or partial results
    after an upgrade. See UPG-8.7.

    The `parsers=` part now reflects ACTUALLY-LOADABLE grammars (via
    available_symbol_languages()) rather than statically-declared ones. This
    means installing or removing a tree-sitter grammar package changes the
    fingerprint and triggers is_stale → rebuild, so a previously grammar-blind
    graph is never silently served after the grammar is installed.

    Reads SYMBOL_SCHEMA_VERSION and available_symbol_languages through the
    package namespace so that test-time monkeypatching of those names on
    agent.symbol_graph is reflected in the fingerprint.
    """
    import hashlib
    import agent.symbol_graph as _sg
    parts = [
        f"schema={_sg.SYMBOL_SCHEMA_VERSION}",
        "parsers=" + ",".join(sorted(_sg.available_symbol_languages())),
        f"embed={embed_model}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def supports_symbols(language: str) -> bool:
    """True if `language` has symbol-graph extraction (locate/trace).

    Normalises common display-name spellings (e.g. "C++"→cpp, "C#"→none) so it
    can be called with either index language keys or human-facing names.
    """
    if not language:
        return False
    norm = language.strip().lower()
    norm = {"c++": "cpp", "cplusplus": "cpp", "objective-c": "c"}.get(norm, norm)
    return norm in SYMBOL_LANGUAGES


# Canonical-ness of a symbol kind for `locate` ranking (UPG-4.5). When several
# symbols share / partially match a name, the user wants "where is X defined",
# so lead with the type/function definition and bury impl blocks and aliases.
# Lower rank = more canonical.
_KIND_RANK: dict[str, int] = {
    "class": 0, "struct": 0, "enum": 0, "interface": 0, "trait": 0,
    "function": 1, "method": 1,
    # UPG-JAVA-ENUM-CONSTANTS-NO-L2-SYMBOLS: enum constants and class
    # fields are members of a type's namespace, ranked like "method" — a
    # class/enum that contains them (rank 0) is the canonical answer when
    # both share a name.
    "field": 1, "enum_constant": 1,
    "route": 2,
    "macro": 3, "variable": 3,
    "impl": 4, "alias": 4, "import": 4,
}
_KIND_RANK_DEFAULT = 2


# Language builtins / stdlib / ubiquitous constructors that pad callee lists with
# noise when answering "what does X call *in this codebase*" (UPG-4.3). A callee
# is treated as a builtin only if it's in this set AND is NOT defined as a symbol
# in the workspace — so a repo that defines its own `len`/`map` keeps it. These
# are suppressed from callee lists by default and shown with include_builtins.
_BUILTINS: dict[str, frozenset[str]] = {
    "python": frozenset({
        "print", "len", "isinstance", "issubclass", "assert", "range", "enumerate",
        "zip", "map", "filter", "sorted", "reversed", "sum", "min", "max", "abs",
        "any", "all", "open", "format", "repr", "str", "int", "float", "bool",
        "list", "dict", "set", "tuple", "frozenset", "bytes", "bytearray",
        "type", "super", "getattr", "setattr", "hasattr", "delattr", "callable",
        "iter", "next", "vars", "dir", "id", "hash", "round", "divmod", "pow",
        "join", "split", "strip", "lstrip", "rstrip", "replace", "startswith",
        "endswith", "lower", "upper", "append", "extend", "pop", "get", "keys",
        "values", "items", "update", "add", "encode", "decode",
    }),
    "rust": frozenset({
        "Ok", "Err", "Some", "None", "Vec", "String", "Box", "Rc", "Arc", "Cell",
        "RefCell", "Mutex", "vec", "format", "println", "print", "eprintln",
        "panic", "assert", "assert_eq", "assert_ne", "write", "writeln", "unwrap",
        "expect", "clone", "into", "from", "to_string", "to_owned", "as_ref",
        "as_mut", "as_str", "borrow", "borrow_mut", "iter", "into_iter", "collect",
        "map", "filter", "push", "pop", "len", "is_empty", "default", "drop",
        "matches", "min", "max", "Default", "Borrowed", "Owned",
    }),
    "go": frozenset({
        "make", "new", "len", "cap", "append", "copy", "delete", "panic",
        "recover", "print", "println", "close", "complex", "real", "imag",
        "string", "byte", "rune", "error", "errors", "fmt",
    }),
    "javascript": frozenset({
        "console", "require", "parseInt", "parseFloat", "isNaN", "JSON",
        "Object", "Array", "String", "Number", "Boolean", "Math", "Date",
        "Promise", "Set", "Map", "Symbol", "Error", "push", "pop", "map",
        "filter", "forEach", "reduce", "slice", "splice", "join", "split",
        "indexOf", "includes", "keys", "values", "entries", "assign",
    }),
    "typescript": frozenset({
        "console", "require", "parseInt", "parseFloat", "isNaN", "JSON",
        "Object", "Array", "String", "Number", "Boolean", "Math", "Date",
        "Promise", "Set", "Map", "Symbol", "Error", "push", "pop", "map",
        "filter", "forEach", "reduce", "slice", "splice", "join", "split",
        "indexOf", "includes", "keys", "values", "entries", "assign",
    }),
    "java": frozenset({
        "System", "String", "Integer", "Long", "Double", "Boolean", "Object",
        "Math", "List", "Map", "Set", "Arrays", "Collections", "Optional",
        "assert", "equals", "hashCode", "toString", "valueOf", "length", "size",
        "get", "add", "put", "remove", "contains", "isEmpty", "println", "print",
    }),
    "c": frozenset({
        "malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset",
        "strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strcat", "strchr",
        "snprintf", "sprintf", "printf", "fprintf", "fputs", "fputc", "puts",
        "abort", "assert", "exit", "sizeof", "offsetof", "va_start", "va_end",
        "va_arg", "qsort", "memcmp",
    }),
    "cpp": frozenset({
        "malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset",
        "strlen", "strcmp", "snprintf", "printf", "fprintf", "abort", "assert",
        "exit", "sizeof", "move", "forward", "make_shared", "make_unique",
        "push_back", "emplace_back", "size", "begin", "end", "at", "find",
        "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast",
    }),
    "zig": frozenset({
        "maxInt", "minInt", "assert", "panic", "print", "alloc", "free", "expect",
        "expectEqual", "expectError", "create", "destroy", "init", "deinit",
        "format", "warn", "debug", "log", "sizeOf", "alignOf", "as", "intCast",
    }),
}

# Call node types per language
_CALL_TYPES: dict[str, set[str]] = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression", "method_call_expression"},
    "java": {"method_invocation", "object_creation_expression"},
    "zig": {"call_expression", "builtin_function"},
    "c": {"call_expression"},
    "cpp": {"call_expression", "new_expression"},
}

# Type-usage node types per language (UPG-4.4). In some languages the dominant
# way code interacts with a type is not a free-function call but a by-value/
# by-reference *usage* — a parameter, return type, field, or generic argument.
# `trace <Type>` was empty for heavily-used Rust types (uv `RegistryClient`,
# `BuildContext`, `PubGrubPackage`) because none of those usages produced an
# edge. We record them as `edge_type="uses"` so the type's call sites surface.
# Keyed by language so it stays opt-in (Rust only for now; extensible later).
_TYPE_USAGE_NODES: dict[str, set[str]] = {
    "rust": {"type_identifier"},
}

# Rust type names we never record a usage edge for: `Self`, std containers, and
# the ubiquitous result/option/collection types. They'd be pure noise to trace
# and would bloat the edge table. Primitives (u32, str, bool, …) are filtered
# separately by the UpperCamelCase convention check in `_record_rust_type`.
_RUST_SKIP_TYPES: frozenset[str] = frozenset({
    "Self", "String", "Vec", "Box", "Rc", "Arc", "Option", "Result", "Cow",
    "Cell", "RefCell", "Mutex", "RwLock", "HashMap", "HashSet", "BTreeMap",
    "BTreeSet", "VecDeque", "Ok", "Err", "Some", "None",
})

# Import node types per language
_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_declaration"},
    "typescript": {"import_declaration"},
    "go": {"import_declaration"},
    "rust": {"use_declaration"},
    "java": {"import_declaration"},
    "zig": {"variable_declaration"},  # const std = @import("std");
    "c": {"preproc_include"},
    "cpp": {"preproc_include"},
}

# Intentionally NOT in config.yaml (Tier-3): _MAX_DEPTH is a recursion
# safety guard tied to Python's frame limit (~1000).  A user who bumped it
# via config could trigger RecursionError on pathological ASTs.
# Guard against pathological ASTs blowing Python's recursion limit. Must be
# counted on EVERY recursion (see below) or it never fires. 200 is far deeper
# than real functions/calls nest, while staying well under Python's ~1000 frame
# limit — deeply-nested data (big C initializer tables) is cut off, not crashed.
_MAX_DEPTH = 200

# ---------------------------------------------------------------------------
# HTTP route extraction regex patterns
# ---------------------------------------------------------------------------

# HTTP method verbs — used to identify route patterns
_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}

# Python decorator patterns: @app.route("/path"), @router.get("/path")
_PY_ROUTE_DECORATOR = re.compile(
    r'@\w+\.(route|' + "|".join(_HTTP_METHODS) + r')\s*\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# Python: method= kwarg on @app.route
_PY_ROUTE_METHOD_KW = re.compile(r'methods\s*=\s*\[([^\]]+)\]', re.IGNORECASE)

# Java Spring annotations
_JAVA_MAPPING = re.compile(
    r'@(Get|Post|Put|Delete|Patch|Request)Mapping\s*\((?:value\s*=\s*)?["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# Express.js: app.get("/path",  router.post("/path",
_EXPRESS_ROUTE = re.compile(
    r'\b(?:app|router|express)\.(get|post|put|delete|patch|use)\s*\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
