"""
T31 — RAGAS retrieval quality evaluation for vectr_search.

Run with: pytest -m ragas tests/test_ragas_eval.py

Requires:
  pip install vectr[ragas]     # ragas + LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)

ragas is NEVER imported at module level. The import happens only inside the
@pytest.mark.ragas-tagged test methods so that the regular test suite (which
excludes the ragas mark) never triggers nest_asyncio.apply() and doesn't
interfere with anyio-backed tests.

The TestDeterministicPrecisionRecall suite always runs and needs no ragas.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal eval dataset
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question: str
    ground_truth: str       # substring expected in a relevant chunk
    expected_category: str  # informational label


def _make_eval_samples() -> list[EvalSample]:
    """Small in-repo eval dataset that doesn't require a large codebase."""
    return [
        EvalSample(
            question="How does vectr_search retrieve results?",
            ground_truth="search",
            expected_category="symbol_definition",
        ),
        EvalSample(
            question="What is the EvictionAdvisor used for?",
            ground_truth="evict",
            expected_category="symbol_callers",
        ),
        EvalSample(
            question="How are notes stored in WorkingContextStore?",
            ground_truth="remember",
            expected_category="cross_session",
        ),
        EvalSample(
            question="What does locate_l2 do when exact match fails?",
            ground_truth="resolution_strategy",
            expected_category="symbol_definition",
        ),
        EvalSample(
            question="How is the MCP turn-count trigger implemented?",
            ground_truth="_session_calls_since_save",
            expected_category="symbol_definition",
        ),
    ]


# ---------------------------------------------------------------------------
# Retrieval harness — real VectrService indexed against vectr source tree
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vectr_service(tmp_path_factory):
    """
    Real VectrService indexed against the vectr source tree with a dummy embedder.
    Uses the same _DummyEmbedProvider pattern as the integration fixtures so no
    model download is required. BM25 lexical search is sufficient for substring-
    based deterministic recall; semantic ranking uses deterministic dummy vectors.

    Yields (service, index_stats) — index_stats is the (files_indexed,
    total_chunks, elapsed_ms) tuple svc.index() returned, kept alongside so
    recall failures can tell an EMPTY corpus (fixture/ignore-rule breakage)
    apart from a ranking collapse.
    """
    from unittest.mock import patch
    from tests.conftest import _DummyEmbedProvider, _RealVectrService

    workspace = str(Path(__file__).parent.parent)
    db_dir = str(tmp_path_factory.mktemp("ragas_eval_db"))

    # The mtime patch below targets the method index_workspace() actually
    # calls (_load_mtime_cache_with_reason); the old patch here targeted the
    # two-name wrapper (_load_mtime_cache) and was therefore INEFFECTIVE — it
    # looked like it forced a full rescan but never ran.
    with patch("agent.indexer.get_embed_provider", return_value=_DummyEmbedProvider()), \
         patch("agent.indexer.CodeIndexer._load_mtime_cache_with_reason", return_value=({}, False)), \
         patch.dict("os.environ", {"VECTR_DB_DIR": db_dir, "VECTR_EMBED_MODEL": "dummy"}):
        # configure_ide=False: this fixture points a real VectrService at the
        # real source tree on purpose (retrieval quality needs real code to
        # search), but indexing the tree must not also REWRITE it. Left at its
        # default True, phase-2 startup writes .cursor/mcp.json,
        # .vscode/mcp.json and .claude/settings.json into the developer's
        # working checkout, pointing their editor at whatever port this
        # in-process service reports. There is no HTTP bind here to advertise.
        svc = _RealVectrService(workspace_root=workspace, configure_ide=False)
        index_stats = svc.index(workspace)
        yield svc, index_stats


def _retrieve_contexts(service, question: str, n: int = 5) -> list[str]:
    """Return top-n chunk content strings from vectr_search."""
    results, _ = service.search(question, n_results=n)
    return [r.content for r in results if hasattr(r, "content")]


# ---------------------------------------------------------------------------
# Deterministic precision/recall — no ragas, always runs in standard CI
# ---------------------------------------------------------------------------

class TestDeterministicPrecisionRecall:
    """
    Lightweight retrieval quality gate that runs in regular CI without ragas.

    deterministic_precision = fraction of retrieved chunks containing ground_truth
    deterministic_recall    = 1 if any retrieved chunk contains ground_truth else 0
    """

    @pytest.fixture(autouse=True)
    def _svc(self, vectr_service):
        self.svc, self.index_stats = vectr_service

    def _corpus_diagnostics(self) -> str:
        """One-line discriminator for recall failures: an empty corpus means
        the FIXTURE broke (ignore rules/path resolution swallowed the tree —
        e.g. UPG-RAGAS-WORKTREE-GUARD's ancestor-exclusion class); a healthy
        corpus with zero recall means ranking collapsed. PYTHONHASHSEED is
        included because the dummy embedder derives vectors from hash() —
        cross-process nondeterminism is a known suspect for flaky recall."""
        files, chunks, elapsed_ms = (list(self.index_stats) + [None, None, None])[:3]
        seed = os.environ.get("PYTHONHASHSEED", "<unset: randomized per process>")
        hint = ""
        if not files:
            hint = " — CORPUS IS EMPTY: the fixture indexed nothing; suspect ignore rules or workspace resolution, NOT ranking"
        return (
            f"indexed_files={files} total_chunks={chunks} index_ms={elapsed_ms} "
            f"PYTHONHASHSEED={seed}{hint}"
        )

    def _evaluate(self, sample: EvalSample, n: int = 5) -> tuple[float, int]:
        contexts = _retrieve_contexts(self.svc, sample.question, n=n)
        if not contexts:
            return 0.0, 0
        hits = sum(1 for c in contexts if sample.ground_truth.lower() in c.lower())
        return hits / len(contexts), 1 if hits > 0 else 0

    def test_search_retrieval_precision_above_zero(self):
        """At least one sample must have non-zero deterministic precision."""
        assert any(
            self._evaluate(s)[0] > 0 for s in _make_eval_samples()
        ), (
            "vectr_search returned 0 relevant chunks for ALL eval samples — "
            f"{self._corpus_diagnostics()}"
        )

    def test_mean_recall_above_threshold(self):
        """Mean recall across all samples must be ≥ 0.4."""
        samples = _make_eval_samples()
        recalls = [self._evaluate(s)[1] for s in samples]
        mean_recall = sum(recalls) / len(recalls)
        assert mean_recall >= 0.4, (
            f"Mean deterministic recall {mean_recall:.2f} < 0.40 — "
            f"per-sample recalls: {recalls}. {self._corpus_diagnostics()}"
        )

    def test_per_sample_results_logged(self):
        """Print a precision/recall table — reporting test, always passes."""
        print("\n--- Vectr retrieval quality (deterministic) ---")
        print(f"{'Category':<20} {'Question':<45} {'P':>5} {'R':>5}")
        print("-" * 80)
        for s in _make_eval_samples():
            p, r = self._evaluate(s)
            short_q = s.question[:43] + ".." if len(s.question) > 45 else s.question
            print(f"{s.expected_category:<20} {short_q:<45} {p:>5.2f} {r:>5}")


# ---------------------------------------------------------------------------
# RAGAS context_precision / context_recall — requires ragas + LLM API key
#
# ragas is imported INSIDE each test method (never at module level) to avoid
# the nest_asyncio.apply() call in ragas/executor.py from polluting the event
# loop used by anyio-backed tests in the same pytest session.
# ---------------------------------------------------------------------------

@pytest.mark.ragas
class TestRagasContextMetrics:
    """
    Full RAGAS evaluation using context_precision and context_recall.

    Requires: pip install vectr[ragas] + OPENAI_API_KEY or ANTHROPIC_API_KEY.
    """

    @pytest.fixture(autouse=True)
    def _svc(self, vectr_service):
        self.svc, self.index_stats = vectr_service

    def _build_ragas_dataset(self, samples: list[EvalSample], n: int = 5):
        datasets = pytest.importorskip("datasets")
        rows = []
        for s in samples:
            contexts = _retrieve_contexts(self.svc, s.question, n=n)
            rows.append({
                "question": s.question,
                "contexts": contexts,
                "ground_truth": s.ground_truth,
                "answer": contexts[0] if contexts else "",
            })
        return datasets.Dataset.from_list(rows)

    def test_context_precision(self):
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            pytest.skip("No LLM API key configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        ragas = pytest.importorskip("ragas")
        from ragas.metrics import context_precision  # noqa: PLC0415
        dataset = self._build_ragas_dataset(_make_eval_samples())
        result = ragas.evaluate(dataset, metrics=[context_precision])
        score = result["context_precision"]
        assert score >= 0.3, (
            f"RAGAS context_precision {score:.3f} < 0.30 — "
            "retrieved chunks are not sufficiently relevant to the questions"
        )

    def test_context_recall(self):
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            pytest.skip("No LLM API key configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        ragas = pytest.importorskip("ragas")
        from ragas.metrics import context_recall  # noqa: PLC0415
        dataset = self._build_ragas_dataset(_make_eval_samples())
        result = ragas.evaluate(dataset, metrics=[context_recall])
        score = result["context_recall"]
        assert score >= 0.3, (
            f"RAGAS context_recall {score:.3f} < 0.30 — "
            "ground-truth information is not present in enough retrieved chunks"
        )
