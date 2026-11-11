"""Codebase fingerprinting and adaptive retrieval strategy selection."""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import EMBEDDING_DEFAULT_MODEL


@dataclass
class CodebaseFingerprint:
    total_files: int
    language_dist: dict[str, int]       # {language: file_count}
    dominant_language: str | None
    is_monorepo: bool
    size_class: str                      # "small" | "medium" | "large"
    doc_coverage_ratio: float = 0.0     # 0.0–1.0 fraction of files with docstrings/comments
    detected_frameworks: list[str] = field(default_factory=list)
    domain_terms: list[str] = field(default_factory=list)
    complexity_class: str = "moderate"  # "simple" | "moderate" | "complex"
    has_grpc: bool = False
    is_legacy: bool = False             # heuristic: very low doc coverage + large + Java/C


@dataclass
class RetrievalStrategy:
    semantic_weight: float      # share of hybrid score from vector search
    bm25_weight: float          # share from BM25 keyword search
    graph_first: bool           # try symbol graph before semantic for locate/trace
    recommended_embed_model: str
    rationale: str

    def __post_init__(self) -> None:
        total = self.semantic_weight + self.bm25_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"semantic_weight + bm25_weight must equal 1.0, got {total}")


_STATIC_LANGS = {"go", "java", "rust"}
_DYNAMIC_LANGS = {"python", "javascript", "typescript"}

_MONOREPO_SIGNALS = [
    "lerna.json", "pnpm-workspace.yaml", "nx.json",
    "build.gradle", "settings.gradle",
    "go.work",
]

# Framework → detection file/pattern
_FRAMEWORK_SIGNALS: dict[str, list[str]] = {
    "django":       ["manage.py", "django"],
    "fastapi":      ["fastapi"],
    "flask":        ["flask"],
    "spring-boot":  ["pom.xml", "spring-boot"],
    "hibernate":    ["hibernate"],
    "gin":          ["gin-gonic/gin"],
    "echo":         ["labstack/echo"],
    "grpc":         [".proto"],
    "react":        ["react", "react-dom"],
    "nextjs":       ["next.config"],
    "vue":          ["vue"],
    "angular":      ["@angular/core"],
    "express":      ["express"],
    "actix":        ["actix-web"],
    "tokio":        ["tokio"],
    "celery":       ["celery"],
}


# ---------------------------------------------------------------------------
# Embed-model autoselection (UPG-EMBEDDER-AUTOSELECT)
# ---------------------------------------------------------------------------
#
# Evidence-keyed embed-model overrides, keyed on
# `CodebaseFingerprint.dominant_language`. SHIPS EMPTY.
#
# The one candidate row this table was built for — a C/C++-dominant
# workspace preferring a different embedder than the granite default —
# was INVALIDATED by the same-index A/B (2026-07-07, cpython corpus,
# identical 21,811-chunk struct-chunk set, dual-vector max-blend, 300-deep
# pool): granite beat arctic on C too (pool-top10 13/24 vs 8/24, top5 8 vs
# 5, in-pool@300 23 vs 19). The earlier C regression that motivated this
# table was mis-attributed to the embedder; it was actually the struct-
# chunking pool-growth + sibling-crowding confound (fixed separately,
# UPG-SIBLING-TYPEDEF-CROWDING). Granite stays the universal default and
# this table has zero live rows — see docs/tasks.md UPG-EMBEDDER-SWAP-
# GRANITE for the full evidence trail.
#
# Adding a row here requires a benchmark VERDICT, not intuition: an A/B run
# on an IDENTICAL chunk set (same chunker/index config — only the embedder
# varies) where the candidate model strictly dominates the current default
# for that `dominant_language`, on a representative multi-case set. A
# single corpus's raw numbers without a same-index control, or a partial
# sample, is not evidence.
EMBED_MODEL_EVIDENCE_TABLE: dict[str, str] = {}


def select_embed_model_from_evidence(
    fp: CodebaseFingerprint,
    table: dict[str, str] | None = None,
) -> str | None:
    """Pure fingerprint -> model-or-None lookup against the evidence table.

    No I/O. `table` defaults to the shipped `EMBED_MODEL_EVIDENCE_TABLE`
    (empty); tests may inject a synthetic table to exercise real selection
    behaviour without shipping a fabricated evidence row. Returns None when
    `fp.dominant_language` has no matching row — including always, while the
    shipped table is empty — so the caller falls back to the configured
    default.
    """
    t = EMBED_MODEL_EVIDENCE_TABLE if table is None else table
    if fp.dominant_language is None:
        return None
    return t.get(fp.dominant_language)


def resolve_embed_model(
    fp: CodebaseFingerprint,
    explicit_override: str | None = None,
    table: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Three-tier embed-model precedence, returning (model, rationale).

    1. `explicit_override` — the caller's already-resolved
       `os.environ.get("VECTR_EMBED_MODEL")` (None when unset) — ALWAYS
       wins; the evidence table is not consulted.
    2. A matching `EMBED_MODEL_EVIDENCE_TABLE` row for this fingerprint wins
       over the built-in default.
    3. No match falls back to the configured default
       (`agent.config.EMBEDDING_DEFAULT_MODEL`).

    Pure and I/O-free: `explicit_override` is passed in rather than read
    from the environment here, so every precedence tier is exercised by
    passing plain arguments — no monkeypatching required.

    `embedding.default_model` (agent/config.yaml) has no "unset" state
    distinct from "vectr's shipped value" — every deployment carries some
    value for it — so it cannot be independently detected as an operator
    override without a fabricated sentinel. It is therefore always tier 3
    ("the configured default"), never a tier-1 override in its own right;
    `VECTR_EMBED_MODEL` is the one channel with genuine unset-vs-set
    semantics today.
    """
    if explicit_override:
        return explicit_override, (
            "embed model explicitly set via VECTR_EMBED_MODEL — evidence "
            f"table not consulted, using {explicit_override}"
        )
    table_model = select_embed_model_from_evidence(fp, table)
    if table_model is not None:
        return table_model, (
            "embed-model evidence row matched dominant language "
            f"{fp.dominant_language!r} — using {table_model}"
        )
    return EMBEDDING_DEFAULT_MODEL, (
        "no embed-model evidence row for this fingerprint — using "
        f"configured default {EMBEDDING_DEFAULT_MODEL}"
    )


def _detect_monorepo(workspace_root: str) -> bool:
    root = Path(workspace_root)
    for signal in _MONOREPO_SIGNALS:
        if (root / signal).exists():
            return True
    pkg_files = {"package.json", "setup.py", "pyproject.toml", "pom.xml", "build.gradle"}
    sub_pkg_count = sum(
        1 for d in root.iterdir()
        if d.is_dir() and any((d / p).exists() for p in pkg_files)
    )
    return sub_pkg_count >= 3


def _tree_has_file(workspace_root: str, name_pattern: str) -> bool:
    """True if any file under workspace_root matches name_pattern, walking only
    the same tree the content indexer would ever see.

    Fingerprinting must never derive a signal from a subtree vectr doesn't
    index — a match under an ignored/fixture directory would skew the
    BM25/semantic strategy blend for a codebase that doesn't actually contain
    that pattern in any indexed file. Prunes the identical exclusion set
    `CodeIndexer.index_workspace` applies (see agent/indexer/_core.py and
    agent/cartographer.py's directory walks): the indexer's own EXCLUDED_DIRS,
    any hidden directory (name starting with "."), and the workspace's
    .vectrignore directory entries — reusing get_vectrignore_dirs rather than
    re-parsing .vectrignore here.
    """
    from agent.indexer import EXCLUDED_DIRS
    from integrations.workspace_detect import get_vectrignore_dirs

    root = Path(workspace_root)
    all_excluded = EXCLUDED_DIRS | get_vectrignore_dirs(str(root))

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in all_excluded and not d.startswith(".")]
        if fnmatch.filter(filenames, name_pattern):
            return True
    return False


def _detect_frameworks(workspace_root: str) -> list[str]:
    root = Path(workspace_root)
    found: list[str] = []

    # Read key dependency files. Fixed root-level filenames only — no
    # directory traversal happens here, so there is no ignored-subtree to
    # honor (unlike the tree walk below).
    dep_text = ""
    for dep_file in ["requirements.txt", "pyproject.toml", "package.json",
                     "pom.xml", "build.gradle", "Cargo.toml", "go.mod"]:
        f = root / dep_file
        if f.exists():
            try:
                dep_text += f.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                pass

    has_proto = _tree_has_file(workspace_root, "*.proto")

    for framework, signals in _FRAMEWORK_SIGNALS.items():
        if framework == "grpc":
            if has_proto:
                found.append(framework)
        elif any(sig in dep_text for sig in signals):
            found.append(framework)

    return found


def _compute_doc_coverage(indexed_files: list[str]) -> float:
    """Fraction of source files that have at least one docstring or comment block."""
    if not indexed_files:
        return 0.0

    _DOC_PATTERNS = re.compile(
        r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|/\*\*[\s\S]*?\*/|^\s*#.+|^\s*//)',
        re.MULTILINE,
    )
    documented = 0
    for fp in indexed_files:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
            if _DOC_PATTERNS.search(text):
                documented += 1
        except OSError:
            pass
    return round(documented / len(indexed_files), 2)


def _extract_domain_terms(workspace_root: str) -> list[str]:
    """Extract recurring capitalised domain nouns from README/top-level docs."""
    root = Path(workspace_root)
    text = ""
    for candidate in ["README.md", "README.rst", "README.txt", "docs/README.md"]:
        f = root / candidate
        if f.exists():
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                break
            except OSError:
                pass
    if not text:
        return []

    # Capitalised words that appear ≥3 times (excluding common English words)
    _STOPWORDS = {
        "The", "This", "That", "With", "From", "When", "Where", "How", "You",
        "Your", "For", "And", "But", "Not", "All", "Are", "Can", "Will", "Has",
        "Use", "Used", "Using", "See", "Run", "Get", "Set", "New", "Each",
        "Into", "After", "Before", "Please", "Also", "More", "Its", "Our",
    }
    words = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', text)
    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= 3][:20]


def _compute_complexity(indexed_files: list[str], dominant_language: str | None) -> str:
    """Estimate complexity from average lines-per-file and nesting depth proxies."""
    if not indexed_files:
        return "moderate"
    total_lines = 0
    sampled = 0
    for fp in indexed_files[:200]:   # sample up to 200 files
        try:
            lines = Path(fp).read_text(encoding="utf-8", errors="ignore").count("\n")
            total_lines += lines
            sampled += 1
        except OSError:
            pass
    if sampled == 0:
        return "moderate"
    avg_lines = total_lines / sampled
    if avg_lines < 100:
        return "simple"
    if avg_lines > 400:
        return "complex"
    return "moderate"


def fingerprint(workspace_root: str, indexed_files: list[str]) -> CodebaseFingerprint:
    """Analyse the workspace and indexed file list to produce a full fingerprint."""
    from agent.indexer import LANG_BY_EXT

    lang_dist: dict[str, int] = {}
    for fp in indexed_files:
        ext = Path(fp).suffix.lower()
        lang = LANG_BY_EXT.get(ext)
        if lang:
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

    total = len(indexed_files)
    dominant = max(lang_dist, key=lang_dist.get) if lang_dist else None

    size_class = "small" if total < 100 else ("large" if total >= 1000 else "medium")

    frameworks = _detect_frameworks(workspace_root)
    doc_ratio = _compute_doc_coverage(indexed_files)
    domain_terms = _extract_domain_terms(workspace_root)
    complexity = _compute_complexity(indexed_files, dominant)

    has_grpc = "grpc" in frameworks
    # Legacy heuristic: large, Java-dominant, very low doc coverage
    is_legacy = (
        size_class in ("medium", "large")
        and dominant == "java"
        and doc_ratio < 0.15
    )

    return CodebaseFingerprint(
        total_files=total,
        language_dist=lang_dist,
        dominant_language=dominant,
        is_monorepo=_detect_monorepo(workspace_root),
        size_class=size_class,
        doc_coverage_ratio=doc_ratio,
        detected_frameworks=frameworks,
        domain_terms=domain_terms,
        complexity_class=complexity,
        has_grpc=has_grpc,
        is_legacy=is_legacy,
    )


def select_strategy(fp: CodebaseFingerprint) -> RetrievalStrategy:
    """Choose retrieval weights and flags from the full codebase fingerprint."""
    reasons: list[str] = []

    # Base weights by size
    if fp.size_class == "small":
        sem, bm = 0.55, 0.45
        reasons.append("small codebase — BM25 weighted higher")
    elif fp.size_class == "large":
        sem, bm = 0.75, 0.25
        reasons.append("large codebase — semantic weighted higher to cut noise")
    else:
        sem, bm = 0.70, 0.30
        reasons.append("medium codebase — balanced hybrid weights")

    graph_first = False

    # Language adjustments
    if fp.dominant_language in _STATIC_LANGS:
        graph_first = True
        reasons.append(f"{fp.dominant_language} — strong static symbol graph, graph traversal first")
    elif fp.dominant_language in _DYNAMIC_LANGS:
        sem = min(sem + 0.05, 0.80)
        bm = round(1.0 - sem, 2)
        reasons.append(f"{fp.dominant_language} dynamic codebase — semantic weight nudged up")

    # Monorepo
    if fp.is_monorepo:
        graph_first = True
        reasons.append("monorepo — symbol graph traversal preferred for cross-module nav")

    # gRPC: proto definitions contain the contract — BM25 on service/method names is precise
    if fp.has_grpc:
        bm = min(bm + 0.10, 0.50)
        sem = round(1.0 - bm, 2)
        graph_first = True
        reasons.append("gRPC services detected — BM25 boosted for proto service/method names")

    # Legacy/compliance: very low doc coverage + large Java — keyword search on formal terms wins
    if fp.is_legacy:
        bm = min(bm + 0.15, 0.55)
        sem = round(1.0 - bm, 2)
        graph_first = True
        reasons.append("legacy Java codebase (low doc coverage) — BM25 heavily weighted for formal term matching")

    # High doc coverage: summaries are rich, semantic search benefits
    if fp.doc_coverage_ratio > 0.70 and not fp.is_legacy:
        sem = min(sem + 0.05, 0.85)
        bm = round(1.0 - sem, 2)
        reasons.append(f"well-documented codebase ({fp.doc_coverage_ratio:.0%} coverage) — semantic weight boosted")

    # Complexity
    if fp.complexity_class == "complex":
        graph_first = True
        reasons.append("complex codebase (high avg file length) — graph traversal preferred")

    # Embed model selection (UPG-EMBEDDER-AUTOSELECT): a deterministic,
    # evidence-keyed table lookup, applied at index time — never a query-side
    # heuristic, since it keys purely on the corpus fingerprint (dominant
    # language) computed above, the same class of index-side signal as every
    # weight/graph_first rule in this function. `EMBED_MODEL_EVIDENCE_TABLE`
    # ships empty (see its module-level comment), so `recommended` resolves
    # to the configured default on every real workspace today.
    recommended, embed_reason = resolve_embed_model(
        fp, explicit_override=os.environ.get("VECTR_EMBED_MODEL")
    )
    reasons.append(embed_reason)

    return RetrievalStrategy(
        semantic_weight=round(sem, 2),
        bm25_weight=round(bm, 2),
        graph_first=graph_first,
        recommended_embed_model=recommended,
        rationale="; ".join(reasons),
    )
