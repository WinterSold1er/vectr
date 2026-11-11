"""
SymbolGraph — L2 knowledge layer: symbols, call graph, import graph.

Enables vectr_locate (where is X defined/used?) and vectr_trace (what calls X?
what does X call?). Built from tree-sitter, no embeddings, no LLM needed.

Design principle: vectr never calls an LLM internally. locate() returns a short
code snippet alongside each symbol so the AI editor can read and understand it
directly — no separate description generation step.

Storage: SQLite in the vectr DB dir. Rebuilt incrementally as files change.

Package layout:
  _types.py      — Symbol, LocateResult, CallEdge dataclasses
  _constants.py  — language maps, regex patterns, constants, metadata functions
  _extraction.py — tree-sitter symbol/edge extraction and HTTP route extraction
  _graph.py      — SQLite-backed SymbolGraph class and locate/trace helpers
"""
from __future__ import annotations

# Re-export the full public API so all existing import sites continue to work:
#   from agent.symbol_graph import <name>

from agent.symbol_graph._types import (
    Symbol,
    LocateResult,
    CallEdge,
)

from agent.symbol_graph._constants import (
    SNIPPET_LINES,
    SYMBOL_SCHEMA_VERSION,
    SYMBOL_LANGUAGES,
    graph_toolchain_fingerprint,
    supports_symbols,
    grammar_available,
    available_symbol_languages,
    # Internal names imported by tests — re-exported so `from agent.symbol_graph
    # import _extract_routes` (test) and `import agent.symbol_graph as _sgmod`
    # (module-level attribute access) continue to resolve.
    _SYMBOL_TYPES,
    _MODULE_BINDING_TYPES,
    _CALL_TYPES,
    _IMPORT_TYPES,
    _TYPE_USAGE_NODES,
    _RUST_SKIP_TYPES,
    _KIND_RANK,
    _KIND_RANK_DEFAULT,
    _BUILTINS,
    _MAX_DEPTH,
    _HTTP_METHODS,
    _PY_ROUTE_DECORATOR,
    _PY_ROUTE_METHOD_KW,
    _JAVA_MAPPING,
    _EXPRESS_ROUTE,
)

from agent.symbol_graph._extraction import (
    extract_symbols_from_file,
    _extract_routes,
    _get_parser,
    _get_symbol_name,
    _get_call_name,
    _module_binding_names,
    _collect_symbols_and_calls,
    _record_rust_type,
    _rust_call_type_head,
    # UPG-BASE-METHOD-OVERRIDE-FLOOD: same-file override-edge extraction.
    # Re-exported so test modules can import it the same way they import
    # the other internal helpers, instead of reaching into
    # agent.symbol_graph._extraction directly.
    _extract_overrides,
    _collect_methods_by_class,
    _python_class_bases,
    _js_class_bases,
)

from agent.symbol_graph._graph import (
    SymbolGraph,
    _levenshtein,
    _get_imported_files,
    _partial_match_key,
    _locate_scope_depth_from_lines,
    _locate_scope_depth_batch,
    _locate_class_enclosed_batch,
    _enclosing_class_from_lines,
    _enclosing_class_from_file,
)

__all__ = [
    # Public dataclasses
    "Symbol",
    "LocateResult",
    "CallEdge",
    # Public constants
    "SNIPPET_LINES",
    "SYMBOL_SCHEMA_VERSION",
    "SYMBOL_LANGUAGES",
    # Public functions
    "graph_toolchain_fingerprint",
    "supports_symbols",
    "grammar_available",
    "available_symbol_languages",
    "extract_symbols_from_file",
    # Public class
    "SymbolGraph",
    # Private names accessed by tests / internal callers
    "_levenshtein",
    "_partial_match_key",
    "_locate_scope_depth_from_lines",
    "_locate_scope_depth_batch",
    "_locate_class_enclosed_batch",
    "_enclosing_class_from_lines",
    "_enclosing_class_from_file",
    "_extract_routes",
    "_get_imported_files",
    "_get_parser",
    "_get_symbol_name",
    "_get_call_name",
    "_module_binding_names",
    "_collect_symbols_and_calls",
    "_record_rust_type",
    "_rust_call_type_head",
    # UPG-BASE-METHOD-OVERRIDE-FLOOD: same-file override extraction
    # and base-list helpers.
    "_extract_overrides",
    "_collect_methods_by_class",
    "_python_class_bases",
    "_js_class_bases",
    "_SYMBOL_TYPES",
    "_MODULE_BINDING_TYPES",
    "_CALL_TYPES",
    "_IMPORT_TYPES",
    "_TYPE_USAGE_NODES",
    "_RUST_SKIP_TYPES",
    "_KIND_RANK",
    "_KIND_RANK_DEFAULT",
    "_BUILTINS",
    "_MAX_DEPTH",
    "_HTTP_METHODS",
    "_PY_ROUTE_DECORATOR",
    "_PY_ROUTE_METHOD_KW",
    "_JAVA_MAPPING",
    "_EXPRESS_ROUTE",
]
