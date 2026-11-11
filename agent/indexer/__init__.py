"""AST-aware code chunking and embedding pipeline.

Package layout:
  _constants.py — LANG_BY_EXT, EXCLUDED_DIRS, batch-size constants
  _types.py     — CodeChunk dataclass, EmbedProvider protocol, embed provider classes
  _chunking.py  — tree-sitter parsers, chunk collection, fallback window chunker
  _core.py      — CodeIndexer orchestration class

All names that existed on the flat agent/indexer.py module are re-exported here
so every existing import site keeps working unchanged:
  from agent.indexer import CodeIndexer        (watcher, searcher, service, tests)
  from agent.indexer import chunk_file         (tests)
  from agent.indexer import LANG_BY_EXT        (watcher, cartographer, symbol_graph)
  from agent.indexer import EXCLUDED_DIRS      (watcher, cartographer)
  from agent.indexer import get_embed_provider (tests — patched at this namespace)
  from agent.indexer import c_symbol_name      (symbol_graph)
  from agent.indexer import _get_parser        (symbol_graph)
  from agent.indexer import _parser_language_for (symbol_graph — UPG-JSFLOW-SYMBOLS)
"""
from __future__ import annotations

# Public constants
from agent.indexer._constants import (
    LANG_BY_EXT,
    EXCLUDED_DIRS,
    _FILE_BATCH_SIZE,
    _EMBED_BATCH_SIZE,
    _UPSERT_BATCH_SIZE,
    _CHUNK_WORKERS,
)

# Public types / embed providers
from agent.indexer._types import (
    CodeChunk,
    EmbedProvider,
    LocalEmbedProvider,
    VoyageEmbedProvider,
    OpenAIEmbedProvider,
    get_embed_provider,
)

# Config-sourced tunables re-exported so `import agent.indexer as m; m._MAX_CHUNK_LINES`
# continues to work (test_config_loader asserts these are the same objects as config.py).
from agent.config import (
    INDEXING_MAX_CHUNK_LINES as _MAX_CHUNK_LINES,
    INDEXING_MAX_CHUNK_CHARS as _MAX_CHUNK_CHARS,
    INDEXING_CLASS_HEADER_LINES as _CLASS_HEADER_LINES,
)

# Public chunking symbols (tests import chunk_file; symbol_graph imports
# _get_parser and c_symbol_name directly from agent.indexer)
from agent.indexer._chunking import (
    chunk_file,
    c_symbol_name,
    _get_parser,
    _PARSER_CACHE,
    _CLASS_NODE_TYPES,
    _CHUNK_NODE_TYPES,
    _SYMBOL_FIELD,
    _C_TYPE_NAME_NODES,
    _c_declarator_name,
    _extract_symbol_name,
    _get_leading_comments,
    _collect_chunks_ast,
    _fallback_window_chunks,
    _chunk_markdown,
    _postprocess_chunks,
    is_flow_javascript,
    _parser_language_for,
)

# Public indexer class
from agent.indexer._core import CodeIndexer

__all__ = [
    # Config-sourced tunables (re-exported for test_config_loader)
    "_MAX_CHUNK_LINES",
    "_MAX_CHUNK_CHARS",
    "_CLASS_HEADER_LINES",
    # Constants
    "LANG_BY_EXT",
    "EXCLUDED_DIRS",
    "_FILE_BATCH_SIZE",
    "_EMBED_BATCH_SIZE",
    "_UPSERT_BATCH_SIZE",
    "_CHUNK_WORKERS",
    # Types / providers
    "CodeChunk",
    "EmbedProvider",
    "LocalEmbedProvider",
    "VoyageEmbedProvider",
    "OpenAIEmbedProvider",
    "get_embed_provider",
    # Chunking
    "chunk_file",
    "c_symbol_name",
    "_get_parser",
    "_PARSER_CACHE",
    "_CLASS_NODE_TYPES",
    "_CHUNK_NODE_TYPES",
    "_SYMBOL_FIELD",
    "_C_TYPE_NAME_NODES",
    "_c_declarator_name",
    "_extract_symbol_name",
    "_get_leading_comments",
    "_collect_chunks_ast",
    "_fallback_window_chunks",
    "_chunk_markdown",
    "_postprocess_chunks",
    "is_flow_javascript",
    "_parser_language_for",
    # Core
    "CodeIndexer",
]
