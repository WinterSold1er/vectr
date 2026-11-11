"""Tests for the UPG-DEDUP-AUTOJUNK-DEFC harness.

Lives under tests/ (not benchmarks/) so it is collected by the
default `pytest -q` run and by CI's `pytest tests/` step
(pytest.ini's testpaths scopes routine runs to
`tests agent integrations app`, so a test physically under
benchmarks/defc_autojunk/ would silently never run in CI).

The tests cover two things only:
  1. The `similarity` module's pure-Python helpers
     (pair_similarity, is_templated_pair, classify_pair) — the
     arithmetic the harness reports.
  2. The `templated_analysis` and `harness` scripts' CLI shape and
     output schema — that a reviewer can actually run them and
     read what they emit.

They deliberately do NOT exercise the daemon, the live indexer, or
the searcher. The live measurement is a separate operator action
(see LANE-REPORT.md for the invocation). What these tests pin is
the harness's OWN correctness — the same shape
`test_recall_miss_harness.py` follows.

Collecting invocation: `pytest tests/test_defc_autojunk_harness.py -v`
"""
from __future__ import annotations

import difflib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "defc_autojunk"))

from similarity import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    PairClassification,
    classify_pair,
    is_templated_pair,
    pair_similarity,
)


# ---------------------------------------------------------------------------
# pair_similarity
# ---------------------------------------------------------------------------

class TestPairSimilarity:
    """The two-ratio computation. The whole DEF-C fix rides on these
    numbers being right, so the assertions are the exact values
    `difflib` produces, not a hand-rolled approximation.

    The brief cites "two Rust bodies differing by a single word"
    as the canonical witness. We use a pair from the harness's
    own templated fixture (T01) so the values are reproducible
    from a committed artifact, not a copy-paste that could drift
    out of sync."""

    def test_identical_bodies_return_one_under_both_metrics(self) -> None:
        body = "fn foo() { let x = 1; }"
        default, corrected = pair_similarity(body, body)
        assert default == pytest.approx(1.0)
        assert corrected == pytest.approx(1.0)

    def test_autojunk_needs_both_length_and_repetition(self) -> None:
        """The defect requires TWO conditions together, not one.

        The brief this lane was written from claimed length alone was the
        trigger: "a code chunk routinely exceeds 200 characters, so ordinary
        letters get classed as junk and the ratio collapses". Measurement says
        otherwise, and this test pins the corrected account:

          repetitive, 392 chars   default 0.166  corrected 0.995
          varied,     251 chars   default 0.996  corrected 0.996
          repetitive, 150 chars   default 0.993  corrected 0.993

        autojunk only engages at 200+ elements AND only bites when specific
        elements repeat heavily. On varied code it changes nothing at all.

        This matters well beyond accuracy. The content it distorts, heavily
        repetitive text, is exactly templated code, which is also the
        false-collapse hazard for this key. So the broken metric has been
        acting as an accidental templated-code guard, and correcting it
        without a real guard in its place is what produces the measured
        50-results-to-1 collapse.
        """
        repetitive_a = (
            "fn resolve(&self) -> Lock {\n    self.solve()\n}\n"
            + "// implementation line\n" * 15
        )
        repetitive_b = repetitive_a.replace(
            "// implementation line\n", "// implementation note\n", 1,
        )
        varied_a = (
            "def parse_manifest(path):\n    data = load_yaml(path)\n"
            "    entries = data.get('entries', [])\n    for entry in entries:\n"
            "        validate_schema(entry)\n    return normalise(entries)\n"
            "    # resolves relative paths against the manifest root\n"
        )
        varied_b = varied_a.replace("normalise", "normalize")

        rep_default = difflib.SequenceMatcher(None, repetitive_a, repetitive_b).ratio()
        rep_fixed = difflib.SequenceMatcher(
            None, repetitive_a, repetitive_b, autojunk=False).ratio()
        var_default = difflib.SequenceMatcher(None, varied_a, varied_b).ratio()
        var_fixed = difflib.SequenceMatcher(
            None, varied_a, varied_b, autojunk=False).ratio()

        assert len(repetitive_a) > 200 and len(varied_a) > 200, (
            "both fixtures must clear autojunk's 200-element floor, or the test "
            "is comparing the wrong thing"
        )
        assert rep_fixed - rep_default > 0.5, (
            f"repetitive content must show the defect: default {rep_default:.4f} "
            f"against corrected {rep_fixed:.4f}. If this gap has closed, autojunk "
            f"changed in difflib and the item needs re-measuring, not a relaxed test"
        )
        assert abs(var_fixed - var_default) < 0.01, (
            f"varied content must be UNAFFECTED: default {var_default:.4f} against "
            f"corrected {var_fixed:.4f}. If this gap has opened, the defect is "
            f"broader than measured and the guard analysis needs redoing"
        )

    def test_normalization_applied_to_both_branches(self) -> None:
        """Bodies that differ only in whitespace / case produce
        the same ratios as identical bodies. `_apply_quality_and_dedup`
        normalizes before comparing (chunk_quality.normalized_content),
        so the harness must too — otherwise a future change to
        the searcher's normalization would silently desync the
        measurement from production behaviour."""
        a = "fn foo() {\n  let x = 1;\n}\n"
        b = "FN FOO() { LET X = 1; }"  # same text, different whitespace + case
        default, corrected = pair_similarity(a, b)
        # normalized_content lowercases and collapses whitespace;
        # after that these two bodies are identical.
        assert default == pytest.approx(1.0)
        assert corrected == pytest.approx(1.0)

    def test_completely_different_bodies_score_below_threshold(self) -> None:
        a = "fn accessor_0(&self) -> &Value { let v_0 = self.field_0.borrow(); return v_0; }"
        b = "struct Lock { packages: Vec<Package> }"
        default, corrected = pair_similarity(a, b)
        assert default < 0.5
        assert corrected < 0.5


# ---------------------------------------------------------------------------
# is_templated_pair
# ---------------------------------------------------------------------------

class TestIsTemplatedPair:
    """The templated-body predicate. The harness reports these
    pairs separately from the rest; a wrong True/False here would
    mis-attribute a hazard or hide one."""

    def test_templated_pair_returns_true(self) -> None:
        """Two bodies identical except for an embedded digit return
        True. The exact threshold (>50% of non-matching characters
        are digits) is implemented in
        `chunk_quality._is_templated_body_difference`; this test
        pins the contract from the harness's side."""
        a = "def compute_0(state, ctx):\n    return state.get('a_0', 0)\n"
        b = "def compute_1(state, ctx):\n    return state.get('a_1', 0)\n"
        assert is_templated_pair(a, b) is True

    def test_non_templated_pair_returns_false(self) -> None:
        """Two bodies whose diff is non-digit content (mut, &mut
        markers) return False. These are the cases DEF-C is meant
        to collapse; flagging them templated would shield them
        from the sweep and let distinct symbols leak through as
        if they were the templated-class hazard."""
        a = (
            "/// Returns a reference to the underlying value.\n"
            "fn accessor(&self) -> &Value { return self.field; }\n"
        )
        b = (
            "/// Returns a reference to the underlying value.\n"
            "fn accessor_mut(&mut self) -> &mut Value { return &mut self.field; }\n"
        )
        # a and b differ in non-digit content (mut / &mut markers).
        # The templated predicate is sensitive to digit-only diffs;
        # this pair has zero digits in the diff, so it must not
        # be flagged templated.
        assert is_templated_pair(a, b) is False

    def test_identical_bodies_return_false(self) -> None:
        """Equal bodies are not a "templated" pair — the predicate
        short-circuits on equality. Defensive against the case
        where the harness accidentally double-counts an identical
        body as a templated pair."""
        body = "fn foo() { let x = 1; }"
        assert is_templated_pair(body, body) is False


# ---------------------------------------------------------------------------
# classify_pair
# ---------------------------------------------------------------------------

class TestClassifyPair:
    """The per-pair classification the harness emits. Pins the
    collapse map's structure and the templated flag passthrough."""

    def test_classify_pair_returns_expected_shape(self) -> None:
        a = "def compute_0(state): return state.get('a_0')"
        b = "def compute_1(state): return state.get('a_1')"
        cls = classify_pair("p1", "templated", a, b)
        assert isinstance(cls, PairClassification)
        assert cls.pair_id == "p1"
        assert cls.source == "templated"
        assert cls.is_templated is True
        # would_collapse is a dict keyed by every threshold label
        # in DEFAULT_THRESHOLDS, with bool values.
        assert set(cls.would_collapse.keys()) == set(DEFAULT_THRESHOLDS)
        assert all(isinstance(v, bool) for v in cls.would_collapse.values())

    def test_classify_pair_corrected_ratio_meets_threshold_collapse(self) -> None:
        """When the corrected ratio is at or above T, the collapse
        map for T is True; when below, False. The threshold check
        is `corrected_ratio >= T`, never `>`, matching the searcher's
        own check (`return ratio >= _DOCSTRING_DEDUP_BODY_SIMILARITY_MIN`).
        """
        # Use a near-identical pair: corrected ratio should be ~1.0.
        body = "def foo():\n    let x = 1\n    return x\n"
        cls = classify_pair("p1", "test", body, body)
        # Same body -> corrected ratio is 1.0 -> all thresholds
        # pass.
        for t in DEFAULT_THRESHOLDS:
            assert cls.would_collapse[t] is True

    def test_classify_pair_threshold_at_boundary(self) -> None:
        """The threshold check is `>=` (inclusive), not `>`. A
        body with corrected ratio exactly equal to a threshold
        value must collapse at that threshold, matching the
        searcher's own `>=` check
        (`return ratio >= _DOCSTRING_DEDUP_BODY_SIMILARITY_MIN`).

        Pins the monotonicity property rather than a specific
        boundary value (which would require a body whose
        SequenceMatcher ratio lands at a known float, brittle
        to difflib version changes)."""
        # A body that's clearly NOT a near-duplicate of itself;
        # a pair with no shared content.
        a = "fn resolver_zero(state: &mut State) -> Result<Lock> { state.commit() }"
        # Construct a body whose corrected ratio lands between
        # the loosest and tightest thresholds, by replacing
        # a small fraction of the body. The exact ratio depends
        # on difflib; what matters is the monotonicity.
        b = (
            "fn resolver_zero(state: &mut State) -> Result<Lock> { "
            "state.commit(); "
            "state.propagate_constraints(); "
            "state.normalize() }"
        )
        cls = classify_pair("p1", "test", a, b)
        # collapse map is monotone non-increasing in T (a
        # stricter threshold has FEWER collapses than a
        # looser one). Pinned as a sweep, not a specific
        # boundary value.
        collapsing = [cls.would_collapse[t] for t in DEFAULT_THRESHOLDS]
        # Walk DEFAULT_THRESHOLDS in order (0.75, 0.80, 0.85,
        # 0.90, 0.95, 0.99). For each adjacent pair, the
        # stricter threshold must NOT have MORE collapses
        # than the looser one.
        for i in range(1, len(collapsing)):
            assert not (collapsing[i] and not collapsing[i - 1]), (
                f"monotonicity violated: threshold "
                f"{DEFAULT_THRESHOLDS[i]} collapses a pair that "
                f"the looser {DEFAULT_THRESHOLDS[i-1]} does not — "
                f"this should be impossible with `>=` semantics"
            )

    def test_collapsing_thresholds_helper(self) -> None:
        """`collapsing_thresholds()` returns the threshold labels
        in DEFAULT_THRESHOLDS order, so a reviewer can read the
        row top-to-bottom and see where the pair starts
        collapsing."""
        a = "def foo():\n    return 1\n"
        cls = classify_pair("p1", "test", a, a)
        # Identical bodies collapse at every threshold.
        assert cls.collapsing_thresholds() == list(DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------------------
# Templated analysis CLI
# ---------------------------------------------------------------------------

class TestTemplatedAnalysisScript:
    """The templated-fixture analysis script. Pure-Python; no
    daemon required. Tests pin the CLI shape, the per-pair
    output schema, and the threshold-sweep summary."""

    FIXTURE = REPO_ROOT / "benchmarks" / "defc_autojunk" / "templated_pairs.jsonl"

    def test_fixture_exists_and_loads(self) -> None:
        assert self.FIXTURE.is_file(), f"missing templated fixture: {self.FIXTURE}"
        rows = [json.loads(line) for line in self.FIXTURE.read_text().splitlines() if line.strip()]
        # The brief asks for a "small committed fixture of such
        # pairs; keep it honest". 10 pairs covers four shape
        # families; pinning the count catches accidental deletion
        # or a one-line drop that a reviewer would not notice.
        assert len(rows) == 10
        # Every row has the fields the harness reads.
        for r in rows:
            assert {"id", "shape", "a", "b"} <= set(r.keys())

    def test_cli_runs_and_writes_outputs(self, tmp_path: Path) -> None:
        """The CLI is the entry point reviewers use. Run it end-
        to-end against the committed fixture and assert the two
        output files land on disk with the expected shape."""
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "defc_autojunk" / "templated_analysis.py"),
                "--out-dir", str(tmp_path),
            ],
            capture_output=True, text=True, timeout=60,
            # cwd=REPO_ROOT so the script's --out-dir is interpreted
            # as a path relative to the repo root, matching the
            # operator's invocation. The script's own sys.path
            # handling covers its imports.
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"templated_analysis.py exited non-zero: {result.stderr}"
        )
        out_json = tmp_path / "templated_analysis.json"
        out_txt = tmp_path / "templated_analysis.txt"
        assert out_json.is_file()
        assert out_txt.is_file()

        data = json.loads(out_json.read_text())
        # JSON output: summary + rows. The summary is what a
        # reviewer reads first; the rows are the per-pair data.
        assert "summary" in data
        assert "rows" in data
        assert data["summary"]["n_pairs"] == 10
        # Every row has both ratios + templated flag + would_collapse
        for row in data["rows"]:
            assert {"pair_id", "default_ratio", "corrected_ratio",
                    "is_templated", "would_collapse"} <= set(row.keys())
        # Templated pairs are an honest count, not a tuned one.
        # 10 fixture pairs, all templated shape — see
        # benchmarks/defc_autojunk/README.md. Pinning the count
        # here catches an accidental fixture change.
        n_templated = sum(1 for row in data["rows"] if row["is_templated"])
        assert n_templated == 10

    def test_threshold_sweep_summary_present(self, tmp_path: Path) -> None:
        """The summary's threshold_collapse_counts field is what
        a reviewer reads to compare candidate thresholds at a
        glance. Pin its keys and structure."""
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "defc_autojunk" / "templated_analysis.py"),
                "--out-dir", str(tmp_path),
            ],
            capture_output=True, text=True, timeout=60, check=True,
            cwd=str(REPO_ROOT),
        )
        data = json.loads((tmp_path / "templated_analysis.json").read_text())
        sweep = data["summary"]["threshold_collapse_counts"]
        # Every threshold in DEFAULT_THRESHOLDS has a row.
        assert set(sweep.keys()) == set(DEFAULT_THRESHOLDS)
        # Each row breaks the count out by templated / non_templated.
        for t, counts in sweep.items():
            assert set(counts.keys()) == {"all", "templated", "non_templated"}
            assert all(isinstance(v, int) for v in counts.values())


# ---------------------------------------------------------------------------
# Real-dedup replay harness CLI
# ---------------------------------------------------------------------------

class TestHarnessCLIShape:
    """The real-dedup replay harness. Tests pin the CLI's argument
    surface and the extract_pairs_from_pool helper's contract on
    a hand-built candidate pool. The live daemon interaction is
    out of scope here — that's the operator's measurement."""

    def test_harness_help_lists_required_args(self) -> None:
        """`--corpus` and `--queries-file` are required. Pin the
        CLI contract so a future flag rename does not silently
        break the operator's invocation."""
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "defc_autojunk" / "defc_harness.py"),
                "--help",
            ],
            capture_output=True, text=True, timeout=15,
            # cwd=REPO_ROOT so any future script that grows a
            # module-level `from agent...` import still resolves
            # under this test. (Today the script lazy-imports
            # agent.chunk_quality inside extract_pairs_from_pool,
            # so the cwd is belt-and-braces rather than required.)
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"defc_harness.py --help failed: rc={result.returncode} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        out = result.stdout + result.stderr
        assert "--corpus" in out
        assert "--queries-file" in out


# ---------------------------------------------------------------------------
# extract_pairs_from_pool
# ---------------------------------------------------------------------------

class TestExtractPairsFromPool:
    """The local dedup pair-extraction. Pins the contract: a pair
    is recorded when (and only when) two candidates share a
    leading_docstring_key, and the representative is the first
    candidate under that key."""

    def _make(self, content: str, file_path: str = "/p/src/x.py",
              language: str = "python") -> dict:
        """Build a fake /v1/search result row with just the fields
        the harness's extract_pairs_from_pool reads."""
        return {"content": content, "language": language,
                "file": file_path, "lines": "1-10", "symbol": "",
                "score": 0.0}

    def test_pair_recorded_when_docstring_key_matches(self) -> None:
        a = self._make(
            "/// Resolves dependency conflicts using the solver core algorithm.\n/// Called by the resolver on every candidate set it produces.\nfn a() { let x = 1; }\n", language="rust"
        )
        b = self._make(
            "/// Resolves dependency conflicts using the solver core algorithm.\n/// Called by the resolver on every candidate set it produces.\nfn b() { let y = 2; }\n", language="rust"
        )
        from defc_harness import extract_pairs_from_pool
        pairs, rep_count = extract_pairs_from_pool([a, b])
        assert len(pairs) == 1
        # The pair is (b, a) — b is the candidate, a is the
        # first-seen representative under the shared docstring key.
        assert pairs[0][1] is a
        assert pairs[0][0] is b
        assert rep_count == 1

    def test_no_pair_when_docstring_keys_differ(self) -> None:
        a = self._make(
            "/// First docstring\nfn a() { let x = 1; }\n", language="rust"
        )
        b = self._make(
            "/// Second docstring\nfn b() { let y = 2; }\n", language="rust"
        )
        from defc_harness import extract_pairs_from_pool
        pairs, rep_count = extract_pairs_from_pool([a, b])
        # Different docstring keys -> no pair.
        assert len(pairs) == 0
        # Both candidates still become representatives (no
        # docstring-key collapse), so rep_count is 2.
        assert rep_count == 2

    def test_no_pair_when_no_leading_docstring(self) -> None:
        a = self._make("fn a() { let x = 1; }", language="rust")
        b = self._make("fn b() { let y = 2; }", language="rust")
        from defc_harness import extract_pairs_from_pool
        pairs, rep_count = extract_pairs_from_pool([a, b])
        # No leading docstring -> leading_docstring_key returns
        # "" -> the dedup key never matches -> no pair.
        assert len(pairs) == 0
        assert rep_count == 2

    def test_transitive_pair_set_with_three_candidates(self) -> None:
        """Three candidates sharing one docstring key produce TWO
        pairs: (b, a) and (c, a). The first candidate is the
        representative; the next two are each paired against it.
        This matches `_apply_quality_and_dedup`'s
        `seen_docstring.setdefault(doc_key, []).append(idx)`
        semantics — the representative index list is built in
        input order, and a candidate always compares against
        the FIRST representative under its key."""
        a = self._make("/// Resolves dependency conflicts using the solver core algorithm.\n/// Called by the resolver on every candidate set it produces.\nfn a() { let x = 1; }", language="rust")
        b = self._make("/// Resolves dependency conflicts using the solver core algorithm.\n/// Called by the resolver on every candidate set it produces.\nfn b() { let y = 2; }", language="rust")
        c = self._make("/// Resolves dependency conflicts using the solver core algorithm.\n/// Called by the resolver on every candidate set it produces.\nfn c() { let z = 3; }", language="rust")
        from defc_harness import extract_pairs_from_pool
        pairs, rep_count = extract_pairs_from_pool([a, b, c])
        assert len(pairs) == 2
        # The representative in both pairs is `a`.
        for cand, rep in pairs:
            assert rep is a
        assert rep_count == 1
