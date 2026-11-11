"""
Tests for strategy_selector — fingerprint() and select_strategy().

fingerprint() reads real files to detect frameworks, languages, and doc
coverage; these tests create minimal tmp_path fixtures rather than loading
the full Django source tree.

select_strategy() is pure logic over a CodebaseFingerprint dataclass,
so it can be tested without any filesystem access.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agent.config import EMBEDDING_DEFAULT_MODEL
from agent.strategy_selector import (
    EMBED_MODEL_EVIDENCE_TABLE,
    CodebaseFingerprint,
    RetrievalStrategy,
    fingerprint,
    resolve_embed_model,
    select_embed_model_from_evidence,
    select_strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp(**kwargs) -> CodebaseFingerprint:
    """Build a CodebaseFingerprint with sensible defaults, override via kwargs."""
    defaults = dict(
        total_files=50,
        language_dist={"python": 50},
        dominant_language="python",
        is_monorepo=False,
        size_class="medium",
        doc_coverage_ratio=0.4,
        detected_frameworks=[],
        domain_terms=[],
        complexity_class="moderate",
        has_grpc=False,
        is_legacy=False,
    )
    defaults.update(kwargs)
    return CodebaseFingerprint(**defaults)


# ---------------------------------------------------------------------------
# CodebaseFingerprint — construction
# ---------------------------------------------------------------------------

class TestCodebaseFingerprint:
    def test_fields_accessible(self) -> None:
        fp = _fp()
        assert fp.total_files == 50
        assert fp.dominant_language == "python"
        assert fp.size_class == "medium"
        assert fp.has_grpc is False
        assert fp.is_legacy is False

    def test_language_dist_stores_counts(self) -> None:
        fp = _fp(language_dist={"python": 30, "javascript": 20})
        assert fp.language_dist["python"] == 30
        assert fp.language_dist["javascript"] == 20


# ---------------------------------------------------------------------------
# RetrievalStrategy — validation
# ---------------------------------------------------------------------------

class TestRetrievalStrategy:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="must equal 1.0"):
            RetrievalStrategy(
                semantic_weight=0.6, bm25_weight=0.6,
                graph_first=False,
                recommended_embed_model="Snowflake/snowflake-arctic-embed-m-v1.5",
                rationale="bad",
            )

    def test_valid_weights_accepted(self) -> None:
        s = RetrievalStrategy(
            semantic_weight=0.70, bm25_weight=0.30,
            graph_first=False,
            recommended_embed_model="Snowflake/snowflake-arctic-embed-m-v1.5",
            rationale="ok",
        )
        assert abs(s.semantic_weight + s.bm25_weight - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# select_strategy — pure logic
# ---------------------------------------------------------------------------

class TestSelectStrategy:
    def test_small_codebase_bm25_higher(self) -> None:
        s = select_strategy(_fp(size_class="small", dominant_language=None))
        assert s.bm25_weight >= 0.40
        assert s.semantic_weight <= 0.60

    def test_large_codebase_semantic_higher(self) -> None:
        s = select_strategy(_fp(size_class="large", dominant_language=None))
        assert s.semantic_weight >= 0.70

    def test_medium_codebase_balanced(self) -> None:
        s = select_strategy(_fp(size_class="medium", dominant_language=None))
        assert 0.60 <= s.semantic_weight <= 0.80

    def test_static_language_enables_graph_first(self) -> None:
        for lang in ("go", "java", "rust"):
            s = select_strategy(_fp(dominant_language=lang))
            assert s.graph_first is True, f"Expected graph_first for {lang}"

    def test_dynamic_language_boosts_semantic(self) -> None:
        baseline = select_strategy(_fp(dominant_language=None, size_class="medium"))
        for lang in ("python", "javascript", "typescript"):
            s = select_strategy(_fp(dominant_language=lang, size_class="medium"))
            assert s.semantic_weight >= baseline.semantic_weight

    def test_monorepo_enables_graph_first(self) -> None:
        s = select_strategy(_fp(is_monorepo=True))
        assert s.graph_first is True

    def test_grpc_boosts_bm25_and_graph_first(self) -> None:
        without = select_strategy(_fp(has_grpc=False))
        with_grpc = select_strategy(_fp(has_grpc=True))
        assert with_grpc.bm25_weight > without.bm25_weight
        assert with_grpc.graph_first is True

    def test_legacy_java_boosts_bm25(self) -> None:
        s = select_strategy(_fp(
            is_legacy=True,
            dominant_language="java",
            size_class="large",
            doc_coverage_ratio=0.05,
        ))
        assert s.bm25_weight >= 0.40
        assert s.graph_first is True

    def test_high_doc_coverage_boosts_semantic(self) -> None:
        low = select_strategy(_fp(doc_coverage_ratio=0.30, is_legacy=False))
        high = select_strategy(_fp(doc_coverage_ratio=0.80, is_legacy=False))
        assert high.semantic_weight >= low.semantic_weight

    def test_complex_codebase_enables_graph_first(self) -> None:
        s = select_strategy(_fp(complexity_class="complex"))
        assert s.graph_first is True

    def test_simple_codebase_no_forced_graph(self) -> None:
        s = select_strategy(_fp(
            complexity_class="simple",
            dominant_language="python",
            is_monorepo=False,
            has_grpc=False,
            is_legacy=False,
        ))
        # graph_first not forced by complexity; may still be True from other factors
        # but dominant_language=python + no monorepo/grpc/legacy → should be False
        assert s.graph_first is False

    def test_all_codebases_recommend_default_model(self, monkeypatch) -> None:
        # select_strategy() reads VECTR_EMBED_MODEL as the tier-1 explicit
        # override (UPG-EMBEDDER-AUTOSELECT); assert this independent of
        # whatever the ambient process environment happens to hold.
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        for lang in ("go", "java", "rust", "python", None):
            s = select_strategy(_fp(dominant_language=lang))
            assert s.recommended_embed_model == EMBEDDING_DEFAULT_MODEL, \
                f"Expected configured default model for lang={lang}, got {s.recommended_embed_model}"

    def test_large_codebase_recommends_default_model(self, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        s = select_strategy(_fp(size_class="large", dominant_language=None))
        assert s.recommended_embed_model == EMBEDDING_DEFAULT_MODEL

    def test_rationale_non_empty(self) -> None:
        s = select_strategy(_fp())
        assert len(s.rationale) > 0

    def test_weights_always_sum_to_one(self) -> None:
        configs = [
            _fp(size_class="small"),
            _fp(size_class="medium"),
            _fp(size_class="large"),
            _fp(dominant_language="go"),
            _fp(is_monorepo=True),
            _fp(has_grpc=True),
            _fp(is_legacy=True, dominant_language="java", size_class="large", doc_coverage_ratio=0.05),
            _fp(doc_coverage_ratio=0.90),
            _fp(complexity_class="complex"),
        ]
        for fp in configs:
            s = select_strategy(fp)
            total = round(s.semantic_weight + s.bm25_weight, 6)
            assert total == 1.0, f"Weights {s.semantic_weight}+{s.bm25_weight}={total} != 1.0"


# ---------------------------------------------------------------------------
# Embed model autoselection — UPG-EMBEDDER-AUTOSELECT
#
# EMBED_MODEL_EVIDENCE_TABLE ships empty (the C→arctic candidate row was
# invalidated by the same-index A/B, see the table's module-level comment) —
# these tests exercise the SELECTION MECHANISM's precedence rules against an
# injected synthetic table so real selection behaviour is proven without
# shipping a fabricated evidence row. Every real workspace resolves through
# the shipped empty table, covered separately by TestSelectStrategy's
# `test_all_codebases_recommend_default_model` / `test_large_codebase_
# recommends_default_model`, which assert byte-identical default selection.
# ---------------------------------------------------------------------------

class TestEmbedModelHonestRationale:
    """TASK 1: the rationale must never assert an optimality vectr has not
    measured. With the shipped table empty, no fingerprint can ever produce
    a table match, so every rationale must describe the no-match/default
    path honestly."""

    def test_no_rationale_claims_optimality(self, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        for lang in ("go", "java", "rust", "python", "javascript", "typescript", "c", None):
            for size in ("small", "medium", "large"):
                s = select_strategy(_fp(dominant_language=lang, size_class=size))
                assert "optimal" not in s.rationale.lower(), (
                    f"rationale asserts unmeasured optimality for lang={lang} size={size}: "
                    f"{s.rationale!r}"
                )

    def test_rationale_states_no_row_matched_and_default_used(self, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        s = select_strategy(_fp(dominant_language="rust", size_class="large"))
        assert "no embed-model evidence row" in s.rationale
        assert EMBEDDING_DEFAULT_MODEL in s.rationale


class TestSelectEmbedModelFromEvidence:
    """Pure fp -> model-or-None lookup (requirement 2). No I/O; behaviour
    proven against an injected synthetic table, never the shipped empty one."""

    def test_empty_shipped_table_never_matches(self) -> None:
        assert EMBED_MODEL_EVIDENCE_TABLE == {}
        for lang in ("c", "python", "rust", None):
            fp = _fp(dominant_language=lang)
            assert select_embed_model_from_evidence(fp) is None

    def test_synthetic_table_match_returns_model(self) -> None:
        fp = _fp(dominant_language="c")
        table = {"c": "some-org/some-c-optimized-embedder"}
        assert select_embed_model_from_evidence(fp, table=table) == "some-org/some-c-optimized-embedder"

    def test_synthetic_table_no_match_returns_none(self) -> None:
        fp = _fp(dominant_language="python")
        table = {"c": "some-org/some-c-optimized-embedder"}
        assert select_embed_model_from_evidence(fp, table=table) is None

    def test_none_dominant_language_never_matches(self) -> None:
        fp = _fp(dominant_language=None)
        table = {"c": "some-org/some-c-optimized-embedder"}
        assert select_embed_model_from_evidence(fp, table=table) is None


class TestResolveEmbedModelPrecedence:
    """Three-tier precedence (requirement 3): explicit override > table row >
    configured default. Each tier tested independently and in combination so
    no tier can silently shadow another."""

    _TABLE = {"c": "some-org/some-c-optimized-embedder"}

    def test_no_override_no_match_falls_back_to_configured_default(self) -> None:
        fp = _fp(dominant_language="python")
        model, rationale = resolve_embed_model(fp, explicit_override=None, table=self._TABLE)
        assert model == EMBEDDING_DEFAULT_MODEL
        assert "no embed-model evidence row" in rationale

    def test_matching_table_row_wins_over_built_in_default(self) -> None:
        fp = _fp(dominant_language="c")
        model, rationale = resolve_embed_model(fp, explicit_override=None, table=self._TABLE)
        assert model == "some-org/some-c-optimized-embedder"
        assert model != EMBEDDING_DEFAULT_MODEL
        assert "evidence row matched" in rationale

    def test_explicit_override_wins_over_matching_table_row(self) -> None:
        fp = _fp(dominant_language="c")
        model, rationale = resolve_embed_model(
            fp, explicit_override="operator/forced-model", table=self._TABLE
        )
        assert model == "operator/forced-model"
        assert "explicitly set" in rationale
        assert "evidence table not consulted" in rationale

    def test_explicit_override_wins_even_with_no_table_match(self) -> None:
        fp = _fp(dominant_language="python")
        model, rationale = resolve_embed_model(
            fp, explicit_override="operator/forced-model", table=self._TABLE
        )
        assert model == "operator/forced-model"

    def test_empty_string_override_treated_as_unset(self) -> None:
        # os.environ.get() returns "" only if the variable is literally set to
        # the empty string — treat that the same as unset (never select an
        # empty model name) rather than short-circuiting to "".
        fp = _fp(dominant_language="c")
        model, _ = resolve_embed_model(fp, explicit_override="", table=self._TABLE)
        assert model == "some-org/some-c-optimized-embedder"

    def test_default_table_param_is_the_shipped_empty_table(self) -> None:
        fp = _fp(dominant_language="c")
        model, rationale = resolve_embed_model(fp, explicit_override=None)
        assert model == EMBEDDING_DEFAULT_MODEL
        assert "no embed-model evidence row" in rationale


class TestSelectStrategyEmbedModelIntegration:
    """select_strategy() is the one production call site (app/service.py's
    `_refresh_strategy`, invoked at index time) that wires the selection
    mechanism through — proven here via monkeypatched env, exactly as a real
    VECTR_EMBED_MODEL-setting operator would trigger it."""

    def test_env_var_override_flows_through_select_strategy(self, monkeypatch) -> None:
        monkeypatch.setenv("VECTR_EMBED_MODEL", "operator/forced-model")
        s = select_strategy(_fp(dominant_language="python"))
        assert s.recommended_embed_model == "operator/forced-model"
        assert "explicitly set" in s.rationale

    def test_no_env_var_uses_default_with_empty_table(self, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        s = select_strategy(_fp(dominant_language="c", size_class="large"))
        assert s.recommended_embed_model == EMBEDDING_DEFAULT_MODEL


class TestSelectStrategyEmbedModelConfigOverlay:
    """`recommended_embed_model` must be the model vectr actually indexes with.
    Reading `VECTR_EMBED_MODEL` alone reported the bundled default for every
    workspace switched by its `.vectr/config.yaml` overlay, so session-status
    contradicted itself (`embed_model` from the overlay, `recommended_embed_model`
    from the env-only path). Same resolver as `app/service.py`, same tiers."""

    @staticmethod
    def _workspace(tmp_path: Path, overlay_model: str | None) -> str:
        if overlay_model is not None:
            cfg = tmp_path / ".vectr"
            cfg.mkdir(parents=True, exist_ok=True)
            (cfg / "config.yaml").write_text(f"embedding:\n  model: {overlay_model}\n")
        return str(tmp_path)

    def test_workspace_overlay_without_env_is_reported(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # no operator ~/.vectr/config.yaml
        s = select_strategy(
            _fp(dominant_language="python"),
            self._workspace(tmp_path, "google/gemini-embedding-2"),
        )
        assert s.recommended_embed_model == "google/gemini-embedding-2"
        assert "explicitly set" in s.rationale

    def test_env_var_beats_the_workspace_overlay(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("VECTR_EMBED_MODEL", "operator/forced-model")
        monkeypatch.setenv("HOME", str(tmp_path))
        s = select_strategy(
            _fp(dominant_language="python"),
            self._workspace(tmp_path, "google/gemini-embedding-2"),
        )
        assert s.recommended_embed_model == "operator/forced-model"

    def test_neither_env_nor_overlay_reports_the_bundled_default(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        monkeypatch.delenv("VECTR_WORKSPACE", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        s = select_strategy(_fp(dominant_language="c"), self._workspace(tmp_path, None))
        assert s.recommended_embed_model == EMBEDDING_DEFAULT_MODEL

    def test_global_config_used_when_the_workspace_has_no_overlay(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        home_cfg = tmp_path / ".vectr"
        home_cfg.mkdir(parents=True, exist_ok=True)
        (home_cfg / "config.yaml").write_text("embedding:\n  model: org/global-model\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        s = select_strategy(_fp(dominant_language="python"), self._workspace(tmp_path, None))
        assert s.recommended_embed_model == "org/global-model"

    def test_workspace_overlay_wins_over_the_global_config_key_by_key(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        home_cfg = tmp_path / ".vectr"
        home_cfg.mkdir(parents=True, exist_ok=True)
        (home_cfg / "config.yaml").write_text("embedding:\n  model: org/global-model\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        s = select_strategy(
            _fp(dominant_language="python"),
            self._workspace(tmp_path, "org/workspace-model"),
        )
        assert s.recommended_embed_model == "org/workspace-model"


# ---------------------------------------------------------------------------
# fingerprint() — filesystem integration
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_empty_workspace_returns_zero_files(self, tmp_path) -> None:
        fp = fingerprint(str(tmp_path), [])
        assert fp.total_files == 0
        assert fp.dominant_language is None
        assert fp.size_class == "small"

    def test_python_files_detected_as_dominant(self, tmp_path) -> None:
        files = []
        for i in range(5):
            f = tmp_path / f"module_{i}.py"
            f.write_text(f"def fn_{i}(): pass\n")
            files.append(str(f))
        fp = fingerprint(str(tmp_path), files)
        assert fp.dominant_language == "python"
        assert fp.language_dist.get("python") == 5

    def test_mixed_language_finds_dominant(self, tmp_path) -> None:
        py_files = []
        for i in range(6):
            f = tmp_path / f"m{i}.py"
            f.write_text("def f(): pass\n")
            py_files.append(str(f))
        js_file = tmp_path / "app.js"
        js_file.write_text("function f() {}\n")
        files = py_files + [str(js_file)]
        fp = fingerprint(str(tmp_path), files)
        assert fp.dominant_language == "python"

    def test_size_class_small(self, tmp_path) -> None:
        files = [str(tmp_path / f"f{i}.py") for i in range(10)]
        for f in files:
            Path(f).write_text("pass\n")
        fp = fingerprint(str(tmp_path), files)
        assert fp.size_class == "small"

    def test_size_class_large(self, tmp_path) -> None:
        files = [str(tmp_path / f"f{i}.py") for i in range(1000)]
        for f in files:
            Path(f).write_text("pass\n")
        fp = fingerprint(str(tmp_path), files)
        assert fp.size_class == "large"

    def test_django_framework_detected(self, tmp_path) -> None:
        (tmp_path / "requirements.txt").write_text("django==4.2\ncelery==5.3\n")
        fp = fingerprint(str(tmp_path), [])
        assert "django" in fp.detected_frameworks

    def test_fastapi_framework_detected(self, tmp_path) -> None:
        (tmp_path / "requirements.txt").write_text("fastapi==0.110.0\nuvicorn\n")
        fp = fingerprint(str(tmp_path), [])
        assert "fastapi" in fp.detected_frameworks

    def test_grpc_detected_from_proto_file(self, tmp_path) -> None:
        proto = tmp_path / "service.proto"
        proto.write_text('syntax = "proto3";\nservice Greeter { rpc SayHello (HelloRequest) returns (HelloReply); }\n')
        fp = fingerprint(str(tmp_path), [])
        assert fp.has_grpc is True
        assert "grpc" in fp.detected_frameworks

    def test_grpc_detected_from_proto_in_real_source_dir(self, tmp_path) -> None:
        # Positive control for the ignore-respecting scan below: a .proto in an
        # ordinary (non-excluded) source subdirectory must still be detected.
        proto_dir = tmp_path / "proto"
        proto_dir.mkdir()
        (proto_dir / "service.proto").write_text('syntax = "proto3";\nservice Greeter {}\n')
        fp = fingerprint(str(tmp_path), [])
        assert fp.has_grpc is True
        assert "grpc" in fp.detected_frameworks

    def test_grpc_not_detected_from_vectrignored_dir(self, tmp_path) -> None:
        # UPG-FINGERPRINT-RESPECT-IGNORE: a .proto under a .vectrignore'd
        # directory is invisible to the content indexer, so it must not
        # produce a phantom gRPC signal that skews the retrieval strategy.
        (tmp_path / ".vectrignore").write_text("fixtures\n")
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "service.proto").write_text('syntax = "proto3";\n')
        fp = fingerprint(str(tmp_path), [])
        assert fp.has_grpc is False
        assert "grpc" not in fp.detected_frameworks

    def test_grpc_not_detected_from_excluded_dirs_entry(self, tmp_path) -> None:
        # A .proto under a directory name in the indexer's own EXCLUDED_DIRS
        # (e.g. node_modules) must not be visible to the fingerprint scan
        # either — same exclusion set the content indexer applies.
        excluded_dir = tmp_path / "node_modules"
        excluded_dir.mkdir()
        (excluded_dir / "vendored.proto").write_text('syntax = "proto3";\n')
        fp = fingerprint(str(tmp_path), [])
        assert fp.has_grpc is False
        assert "grpc" not in fp.detected_frameworks

    def test_grpc_not_detected_from_hidden_dir(self, tmp_path) -> None:
        # A .proto under any hidden directory (name starting with ".") must
        # not be visible either — the content indexer never descends into
        # hidden directories.
        hidden_dir = tmp_path / ".fixtures"
        hidden_dir.mkdir()
        (hidden_dir / "service.proto").write_text('syntax = "proto3";\n')
        fp = fingerprint(str(tmp_path), [])
        assert fp.has_grpc is False
        assert "grpc" not in fp.detected_frameworks

    def test_doc_coverage_ratio_zero_for_no_docs(self, tmp_path) -> None:
        files = []
        for i in range(3):
            f = tmp_path / f"nodoc_{i}.py"
            f.write_text(f"def fn_{i}(): x = 1\n")
            files.append(str(f))
        fp = fingerprint(str(tmp_path), files)
        assert fp.doc_coverage_ratio == 0.0

    def test_doc_coverage_ratio_for_commented_files(self, tmp_path) -> None:
        files = []
        for i in range(4):
            f = tmp_path / f"doc_{i}.py"
            f.write_text(f'"""Module {i}."""\ndef fn(): pass\n')
            files.append(str(f))
        fp = fingerprint(str(tmp_path), files)
        assert fp.doc_coverage_ratio == 1.0

    def test_domain_terms_extracted_from_readme(self, tmp_path) -> None:
        readme = tmp_path / "README.md"
        readme.write_text(
            "# Auction Platform\n\n"
            "Auction handles Bidding. Bidding occurs in Segments. "
            "Segments contain AuctionItems. Bidding triggers Auction results. "
            "Auction, Bidding, Segments — these are core to AuctionPlatform.\n"
        )
        fp = fingerprint(str(tmp_path), [])
        # "Auction" and "Bidding" appear ≥3 times each
        assert any(t in ("Auction", "Bidding", "Segments") for t in fp.domain_terms)

    def test_no_readme_domain_terms_empty(self, tmp_path) -> None:
        fp = fingerprint(str(tmp_path), [])
        assert fp.domain_terms == []

    def test_monorepo_detected_from_go_work(self, tmp_path) -> None:
        (tmp_path / "go.work").write_text("go 1.21\n")
        fp = fingerprint(str(tmp_path), [])
        assert fp.is_monorepo is True

    def test_monorepo_detected_from_subdirectory_packages(self, tmp_path) -> None:
        for pkg in ("serviceA", "serviceB", "serviceC"):
            d = tmp_path / pkg
            d.mkdir()
            (d / "setup.py").write_text("from setuptools import setup; setup(name='x')\n")
        fp = fingerprint(str(tmp_path), [])
        assert fp.is_monorepo is True

    def test_non_monorepo_single_package(self, tmp_path) -> None:
        (tmp_path / "setup.py").write_text("from setuptools import setup; setup(name='x')\n")
        fp = fingerprint(str(tmp_path), [])
        assert fp.is_monorepo is False

    def test_complexity_simple_for_tiny_files(self, tmp_path) -> None:
        files = []
        for i in range(5):
            f = tmp_path / f"tiny_{i}.py"
            f.write_text("x = 1\n")
            files.append(str(f))
        fp = fingerprint(str(tmp_path), files)
        assert fp.complexity_class == "simple"

    def test_complexity_complex_for_large_files(self, tmp_path) -> None:
        files = []
        for i in range(3):
            f = tmp_path / f"big_{i}.py"
            # 500 lines each → avg > 400
            f.write_text("\n".join(f"x_{j} = {j}" for j in range(500)))
            files.append(str(f))
        fp = fingerprint(str(tmp_path), files)
        assert fp.complexity_class == "complex"

    def test_legacy_heuristic_java_low_doc_large(self, tmp_path) -> None:
        files = []
        for i in range(50):
            f = tmp_path / f"Class{i}.java"
            f.write_text(f"public class Class{i} {{}}\n")  # no javadoc
            files.append(str(f))
        # Simulate "large" by passing 1000 hypothetical paths (only 50 exist for coverage)
        fake_files = files + [str(tmp_path / f"Fake{i}.java") for i in range(950)]
        fp = fingerprint(str(tmp_path), fake_files)
        assert fp.dominant_language == "java"
        assert fp.size_class == "large"
        assert fp.doc_coverage_ratio < 0.15
        assert fp.is_legacy is True


# ---------------------------------------------------------------------------
# Multi-signal scenarios — realistic codebase archetypes
# ---------------------------------------------------------------------------

class TestMultiSignalScenarios:
    def test_large_java_grpc_legacy_monorepo(self) -> None:
        # large: bm=0.25; gRPC: min(0.35,0.50)=0.35; legacy: min(0.50,0.55)=0.50
        s = select_strategy(_fp(
            size_class="large",
            dominant_language="java",
            has_grpc=True,
            is_legacy=True,
            is_monorepo=True,
            doc_coverage_ratio=0.05,
        ))
        assert s.bm25_weight <= 0.55
        assert s.graph_first is True
        assert abs(s.semantic_weight + s.bm25_weight - 1.0) < 1e-6

    def test_small_python_fastapi(self) -> None:
        # small: bm=0.45; python nudge: sem=0.60, bm=0.40 — BM25 stays high.
        s = select_strategy(_fp(
            size_class="small",
            dominant_language="python",
            detected_frameworks=["fastapi"],
            doc_coverage_ratio=0.2,
        ))
        assert s.bm25_weight >= 0.40
        assert s.graph_first is False
        assert abs(s.semantic_weight + s.bm25_weight - 1.0) < 1e-6

    def test_medium_typescript_react(self) -> None:
        # medium: sem=0.70; typescript nudge: sem=min(0.75,0.80)=0.75
        s = select_strategy(_fp(
            size_class="medium",
            dominant_language="typescript",
            detected_frameworks=["react"],
        ))
        assert s.semantic_weight >= 0.75
        assert s.graph_first is False
        assert abs(s.semantic_weight + s.bm25_weight - 1.0) < 1e-6

    def test_large_go_monorepo(self) -> None:
        # large + static Go + monorepo — both trigger graph_first, no weight conflict.
        s = select_strategy(_fp(
            size_class="large",
            dominant_language="go",
            is_monorepo=True,
        ))
        assert s.graph_first is True
        assert s.semantic_weight >= 0.75
        assert abs(s.semantic_weight + s.bm25_weight - 1.0) < 1e-6

    def test_small_rust_cli(self) -> None:
        # small (BM25 favoured) + Rust (static, graph_first).
        s = select_strategy(_fp(
            size_class="small",
            dominant_language="rust",
            doc_coverage_ratio=0.4,
        ))
        assert s.graph_first is True
        assert s.bm25_weight >= 0.40
        assert abs(s.semantic_weight + s.bm25_weight - 1.0) < 1e-6

    def test_well_documented_python_django(self) -> None:
        # medium: sem=0.70; python: sem=0.75; high_doc: sem=min(0.80,0.85)=0.80
        s = select_strategy(_fp(
            size_class="medium",
            dominant_language="python",
            detected_frameworks=["django"],
            doc_coverage_ratio=0.85,
            is_legacy=False,
        ))
        assert s.semantic_weight == 0.80
        assert s.bm25_weight == 0.20
        assert s.graph_first is False

    def test_medium_java_grpc_legacy_bm25_capped(self) -> None:
        # medium: bm=0.30; gRPC: min(0.40,0.50)=0.40; legacy: min(0.55,0.55)=0.55 — cap hit.
        s = select_strategy(_fp(
            size_class="medium",
            dominant_language="java",
            has_grpc=True,
            is_legacy=True,
            doc_coverage_ratio=0.05,
        ))
        assert s.bm25_weight == 0.55
        assert s.semantic_weight == 0.45
        assert s.graph_first is True

    def test_large_python_high_doc_semantic_capped(self) -> None:
        # large: sem=0.75; python: min(0.80,0.80)=0.80; high_doc: min(0.85,0.85)=0.85 — cap hit.
        s = select_strategy(_fp(
            size_class="large",
            dominant_language="python",
            doc_coverage_ratio=0.85,
            is_legacy=False,
        ))
        assert s.semantic_weight == 0.85
        assert s.bm25_weight == 0.15
        assert s.graph_first is False


# ---------------------------------------------------------------------------
# Weight bound enforcement — caps are never violated
# ---------------------------------------------------------------------------

class TestWeightBounds:
    def test_bm25_grpc_cap_0_50_hit_for_small(self) -> None:
        # small: bm=0.45; gRPC raw boost = 0.55 → capped at 0.50.
        s = select_strategy(_fp(size_class="small", has_grpc=True, dominant_language=None))
        assert s.bm25_weight == 0.50
        assert s.semantic_weight == 0.50

    def test_bm25_legacy_cap_0_55_hit(self) -> None:
        # medium+gRPC+legacy: bm=0.30→0.40→cap at 0.55.
        s = select_strategy(_fp(
            size_class="medium",
            dominant_language="java",
            has_grpc=True,
            is_legacy=True,
            doc_coverage_ratio=0.05,
        ))
        assert s.bm25_weight == 0.55

    def test_bm25_never_exceeds_0_55(self) -> None:
        combos = [
            _fp(size_class="small", has_grpc=True, is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="medium", has_grpc=True, is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="large", has_grpc=True, is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="small", is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="medium", is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
        ]
        for fp in combos:
            s = select_strategy(fp)
            assert s.bm25_weight <= 0.55, f"BM25={s.bm25_weight} > 0.55 for {fp}"

    def test_semantic_dynamic_cap_0_80_hit_for_large_python(self) -> None:
        # large: sem=0.75; python: min(0.80,0.80)=0.80 — cap hit exactly.
        s = select_strategy(_fp(size_class="large", dominant_language="python"))
        assert s.semantic_weight == 0.80
        assert s.bm25_weight == 0.20

    def test_semantic_doc_coverage_cap_0_85_hit(self) -> None:
        # large+python+high_doc: 0.75→0.80→0.85 — absolute cap.
        s = select_strategy(_fp(
            size_class="large",
            dominant_language="python",
            doc_coverage_ratio=0.90,
            is_legacy=False,
        ))
        assert s.semantic_weight == 0.85
        assert s.bm25_weight == 0.15

    def test_semantic_never_exceeds_0_85(self) -> None:
        combos = [
            _fp(size_class="large", dominant_language="python", doc_coverage_ratio=0.95),
            _fp(size_class="large", dominant_language="javascript", doc_coverage_ratio=0.95),
            _fp(size_class="large", dominant_language="typescript", doc_coverage_ratio=0.95),
            _fp(size_class="medium", dominant_language="python", doc_coverage_ratio=0.95),
        ]
        for fp in combos:
            s = select_strategy(fp)
            assert s.semantic_weight <= 0.85, f"semantic={s.semantic_weight} > 0.85 for {fp}"

    def test_weights_sum_to_one_comprehensive(self) -> None:
        configs = [
            _fp(size_class="small", dominant_language=None),
            _fp(size_class="small", dominant_language="python"),
            _fp(size_class="small", dominant_language="go"),
            _fp(size_class="small", has_grpc=True),
            _fp(size_class="small", is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="small", has_grpc=True, is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="medium", dominant_language=None),
            _fp(size_class="medium", dominant_language="typescript"),
            _fp(size_class="medium", dominant_language="go", is_monorepo=True),
            _fp(size_class="medium", has_grpc=True),
            _fp(size_class="medium", is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="medium", has_grpc=True, is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="medium", doc_coverage_ratio=0.85),
            _fp(size_class="medium", dominant_language="python", doc_coverage_ratio=0.85),
            _fp(size_class="large", dominant_language=None),
            _fp(size_class="large", dominant_language="rust"),
            _fp(size_class="large", dominant_language="python"),
            _fp(size_class="large", dominant_language="python", doc_coverage_ratio=0.85),
            _fp(size_class="large", dominant_language="go", is_monorepo=True),
            _fp(size_class="large", has_grpc=True),
            _fp(size_class="large", is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="large", has_grpc=True, is_legacy=True, dominant_language="java", doc_coverage_ratio=0.05),
            _fp(size_class="large", complexity_class="complex", dominant_language="go", is_monorepo=True),
        ]
        for fp in configs:
            s = select_strategy(fp)
            total = round(s.semantic_weight + s.bm25_weight, 6)
            assert total == 1.0, f"{s.semantic_weight}+{s.bm25_weight}={total} != 1.0 for {fp}"


# ---------------------------------------------------------------------------
# Rationale completeness — every activated signal leaves a keyword trace
# ---------------------------------------------------------------------------

class TestRationaleKeywords:
    def test_rationale_mentions_small(self) -> None:
        s = select_strategy(_fp(size_class="small"))
        assert "small" in s.rationale.lower()

    def test_rationale_mentions_medium(self) -> None:
        s = select_strategy(_fp(size_class="medium"))
        assert "medium" in s.rationale.lower()

    def test_rationale_mentions_large(self) -> None:
        s = select_strategy(_fp(size_class="large"))
        assert "large" in s.rationale.lower()

    def test_rationale_mentions_static_language(self) -> None:
        for lang in ("go", "java", "rust"):
            s = select_strategy(_fp(dominant_language=lang))
            assert lang in s.rationale.lower(), f"Expected '{lang}' in rationale"

    def test_rationale_mentions_dynamic_language(self) -> None:
        for lang in ("python", "javascript", "typescript"):
            s = select_strategy(_fp(dominant_language=lang))
            assert lang in s.rationale.lower(), f"Expected '{lang}' in rationale"

    def test_rationale_mentions_monorepo(self) -> None:
        s = select_strategy(_fp(is_monorepo=True))
        assert "monorepo" in s.rationale.lower()

    def test_rationale_mentions_grpc(self) -> None:
        s = select_strategy(_fp(has_grpc=True))
        assert "grpc" in s.rationale.lower()

    def test_rationale_mentions_legacy(self) -> None:
        s = select_strategy(_fp(is_legacy=True, dominant_language="java"))
        assert "legacy" in s.rationale.lower()

    def test_rationale_mentions_high_doc_coverage(self) -> None:
        s = select_strategy(_fp(doc_coverage_ratio=0.85, is_legacy=False))
        assert "documented" in s.rationale.lower() or "coverage" in s.rationale.lower()

    def test_rationale_mentions_complex(self) -> None:
        s = select_strategy(_fp(complexity_class="complex"))
        assert "complex" in s.rationale.lower()


# ---------------------------------------------------------------------------
# Real workspace fingerprint — vectr repo itself (no model download needed)
# ---------------------------------------------------------------------------

class TestFingerprintRealWorkspace:
    @classmethod
    def _collect_py_files(cls) -> list[str]:
        root = Path(__file__).parent.parent
        # UPG-TESTFP-TMP: `tmp/` holds benchmark/acceptance fixtures (e.g. a full
        # django checkout, 2,900+ .py files) and `.claude/`/`.worktrees/` hold
        # coder worktrees — none reflects vectr's own repo shape, so all inflate
        # this self-fingerprint test's file count out of the asserted size-class range.
        #
        # The skip check must look at path parts RELATIVE to `root`, not the
        # absolute path: this suite itself often runs from inside a worktree
        # physically nested under a directory named `.claude`
        # (`.claude/worktrees/<name>/`), so checking absolute parts would make
        # a bare `.claude` skip-entry exclude every file under `root` — an
        # accidental match on an ancestor directory name that has nothing to
        # do with the workspace being fingerprinted.
        skip = {".venv", "__pycache__", ".git", ".pytest_cache", "tmp", ".claude", ".worktrees"}
        return [
            str(f)
            for f in root.rglob("*.py")
            if not any(part in skip for part in f.relative_to(root).parts)
        ]

    def test_dominant_language_is_python(self) -> None:
        fp = fingerprint(str(Path(__file__).parent.parent), self._collect_py_files())
        assert fp.dominant_language == "python"

    def test_fastapi_detected(self) -> None:
        fp = fingerprint(str(Path(__file__).parent.parent), self._collect_py_files())
        assert "fastapi" in fp.detected_frameworks

    def test_not_legacy(self) -> None:
        fp = fingerprint(str(Path(__file__).parent.parent), self._collect_py_files())
        assert fp.is_legacy is False

    def test_strategy_graph_first_tracks_complexity_only(self) -> None:
        # vectr is pure dynamic Python — not a monorepo, no gRPC, not legacy — so the
        # ONLY trigger that can set graph_first for this repo is complexity_class.
        # (As the repo grew, avg file length crossed the 400-line "complex" threshold,
        # so this is now True; asserting the relationship keeps the test robust to size.)
        fp = fingerprint(str(Path(__file__).parent.parent), self._collect_py_files())
        assert select_strategy(fp).graph_first is (fp.complexity_class == "complex")

    def test_size_class_small_or_medium(self) -> None:
        fp = fingerprint(str(Path(__file__).parent.parent), self._collect_py_files())
        assert fp.size_class in {"small", "medium"}
