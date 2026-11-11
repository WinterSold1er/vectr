"""Tests for agent/outcome.py (memoization-l1-capture-design §2.4).

The outcome-derivation cascade: content markers over tool OUTPUT (never
prompt content — R5-sanctioned) beat the exit code, which beats the weak
is_error/interrupted fallback, which beats "unknown". Markers are primary
because the exit code lies in the common case (T2 research finding: 0
observed BUILD FAILURE exit codes across 25 real maven invocations)."""
from __future__ import annotations

from agent.outcome import derive_outcome, derive_termination, match_markers


class TestMarkersBeatExitCode:
    def test_failure_marker_with_rc_zero_is_soft_failure(self):
        result = derive_outcome(
            rc=0, is_error=False, interrupted=False,
            stdout_digest="[INFO] BUILD FAILURE", stderr_digest="",
        )
        assert result["outcome"] == "soft_failure"
        assert "maven.build_failure" in result["markers_matched"]

    def test_failure_marker_with_no_rc_is_soft_failure(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="BUILD FAILURE", stderr_digest="",
        )
        assert result["outcome"] == "soft_failure"

    def test_failure_marker_with_nonzero_rc_is_failure(self):
        result = derive_outcome(
            rc=1, is_error=False, interrupted=False,
            stdout_digest="BUILD FAILURE", stderr_digest="",
        )
        assert result["outcome"] == "failure"

    def test_success_marker_with_no_rc_is_success(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="5 passed in 1.2s", stderr_digest="",
        )
        assert result["outcome"] == "success"


class TestExitCodeFallback:
    def test_rc_zero_no_markers_is_success(self):
        result = derive_outcome(
            rc=0, is_error=False, interrupted=False,
            stdout_digest="ok", stderr_digest="",
        )
        assert result["outcome"] == "success"

    def test_rc_nonzero_no_markers_is_failure(self):
        result = derive_outcome(
            rc=1, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
        )
        assert result["outcome"] == "failure"


class TestHarnessSuccessTier:
    """UPG-EPISODE-OUTCOME-CLASSIFICATION: `harness_success` (tri-state,
    caller-supplied only when it genuinely knows which discrete hook event
    routed the tool call — the same tool-call-structure category as `rc`)
    fills the tier between `is_error` and `unknown`, closing the gap where a
    successfully-routed everyday command carries no exit code and matches no
    build/test marker. It sits BELOW `is_error`, not above it: `is_error`,
    when an integration populates it, is a direct per-call truth claim,
    while `harness_success` is a routing inference that only holds under an
    assumed event vocabulary (not guaranteed for every integration — see
    agent/outcome.py's module docstring)."""

    def test_harness_success_true_no_rc_no_markers_is_success(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="{}\n", stderr_digest="",
            harness_success=True,
        )
        assert result["outcome"] == "success"

    def test_harness_success_false_no_rc_no_markers_is_failure(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
            harness_success=False,
        )
        assert result["outcome"] == "failure"

    def test_harness_success_none_no_rc_no_markers_stays_unknown(self):
        """The default: an integration that doesn't supply the signal gets
        exactly the pre-fix behavior — never guessed."""
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
            harness_success=None,
        )
        assert result["outcome"] == "unknown"

    def test_rc_still_beats_harness_success_when_both_present(self):
        result = derive_outcome(
            rc=1, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
            harness_success=True,
        )
        assert result["outcome"] == "failure"

    def test_failure_marker_beats_harness_success_true(self):
        """The T2 exit-code-lying case, restated for harness_success: a
        build tool piped through a display-only stage can report
        harness_success=True (the harness itself saw a clean top-level
        exit) while its own printed output says otherwise — markers still
        win, producing soft_failure, never a false success."""
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="[INFO] BUILD FAILURE", stderr_digest="",
            harness_success=True,
        )
        assert result["outcome"] == "soft_failure"

    def test_success_marker_beats_harness_success_false(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="5 passed in 1.2s", stderr_digest="",
            harness_success=False,
        )
        assert result["outcome"] == "success"

    def test_interrupted_beats_harness_success_true(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=True,
            stdout_digest="", stderr_digest="",
            harness_success=True,
        )
        assert result["outcome"] == "interrupted"

    def test_is_error_true_beats_harness_success_true(self):
        """`is_error`, when the caller supplies True, wins over a
        conflicting `harness_success=True` (agent/outcome.py's derive_outcome
        docstring): `is_error` is a direct per-call truth claim, while
        `harness_success` is only a routing inference that assumes the
        editor's event vocabulary cleanly separates success from failure by
        event name — an assumption not guaranteed for every integration
        (e.g. one whose hook surface fires the same event name regardless of
        outcome). Never observed to disagree on Claude Code today (G0:
        `is_error` is structurally always False on its verified success
        path, so this tier never fires there)."""
        result = derive_outcome(
            rc=None, is_error=True, interrupted=False,
            stdout_digest="", stderr_digest="",
            harness_success=True,
        )
        assert result["outcome"] == "failure"

    def test_is_error_true_still_used_when_harness_success_is_none(self):
        result = derive_outcome(
            rc=None, is_error=True, interrupted=False,
            stdout_digest="", stderr_digest="",
            harness_success=None,
        )
        assert result["outcome"] == "failure"

    def test_default_omitted_harness_success_matches_none(self):
        """Callers that don't pass harness_success at all (existing
        integrations, pre-fix call sites) get identical behavior to
        harness_success=None explicitly — additive, not a breaking change."""
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
        )
        assert result["outcome"] == "unknown"


class TestSignalTermination:
    def test_rc_137_is_interrupted_and_signal(self):
        result = derive_outcome(
            rc=137, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
        )
        assert result["outcome"] == "interrupted"
        assert result["termination"] == "signal"


class TestWeakFallbacks:
    def test_is_error_flag_no_rc_no_markers(self):
        result = derive_outcome(
            rc=None, is_error=True, interrupted=False,
            stdout_digest="", stderr_digest="",
        )
        assert result["outcome"] == "failure"

    def test_interrupted_flag_no_rc_no_markers(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=True,
            stdout_digest="", stderr_digest="",
        )
        assert result["outcome"] == "interrupted"
        assert result["termination"] == "cancelled"

    def test_nothing_present_is_unknown(self):
        result = derive_outcome(
            rc=None, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
        )
        assert result["outcome"] == "unknown"
        assert result["termination"] == "unknown"


class TestTerminationDerivation:
    def test_normal_exit(self):
        assert derive_termination(0, False) == "normal"

    def test_no_rc_no_interrupted(self):
        assert derive_termination(None, False) == "unknown"

    def test_interrupted_wins_over_rc(self):
        assert derive_termination(0, True) == "cancelled"

    def test_signal_range_rc(self):
        assert derive_termination(137, False) == "signal"


class TestMarkerMatchingOverTextOnly:
    def test_matches_in_stderr_too(self):
        matched = match_markers("", "error: could not compile")
        assert any(kind == "failure" for _, kind in matched)

    def test_no_match_returns_empty(self):
        assert match_markers("nothing interesting here", "") == []


class TestGoFailMarkerDoesNotClaimPytestFailures:
    """`go_test.fail` was a bare `^FAIL`, and pytest's short summary prints
    one `FAILED tests/x.py::test_y` line per failing test. Because
    `match_markers` scans the combined digest with no per-tool scoping, every
    failing Python run also matched the Go marker. Both are kind: failure, so
    the OUTCOME was right by accident — but `markers_matched` is the record of
    which tool reported what, and arcs are keyed off it, so a Go marker on a
    pytest run is a false attribution. Same cross-tool contamination class as
    TestZeroCountNeverFalsePositivesAsFailure below."""

    def test_pytest_failed_summary_does_not_match_the_go_marker(self):
        result = derive_outcome(
            rc=1, is_error=False, interrupted=False,
            stdout_digest=(
                "FAILED tests/test_x.py::test_a - assert 1 == 2\n"
                "1 failed, 5 passed in 0.42s"
            ),
            stderr_digest="",
        )
        assert result["outcome"] == "failure"
        assert "pytest.failed" in result["markers_matched"]
        assert "go_test.fail" not in result["markers_matched"]

    def test_real_go_failure_still_matches(self):
        matched = [m for m, _ in match_markers("FAIL\tgithub.com/x/y\t0.02s", "")]
        assert "go_test.fail" in matched

    def test_go_build_failure_form_still_matches(self):
        matched = [m for m, _ in match_markers("FAIL\tgithub.com/x/y [build failed]", "")]
        assert "go_test.fail" in matched

    def test_bare_fail_line_still_matches(self):
        matched = [m for m, _ in match_markers("some output\nFAIL", "")]
        assert "go_test.fail" in matched


class TestZeroCountNeverFalsePositivesAsFailure:
    """Adversarial-review fix B3: `\\d+ failed`-style patterns match a fully
    GREEN run's own "0 failed" summary line. `match_markers` scans the
    combined stdout+stderr digest with no per-tool scoping (agent/outcome.py
    docstring), so this is also a cross-tool risk — a green cargo test
    summary containing "0 failed" must never trip the pytest/jest markers.
    `maven.tests_failed` already used the `[1-9]` convention this table now
    matches everywhere a "N failed"-shaped marker exists."""

    def test_pytest_zero_failed_summary_is_success(self):
        result = derive_outcome(
            rc=0, is_error=False, interrupted=False,
            stdout_digest="5 passed, 0 failed in 1.2s", stderr_digest="",
        )
        assert result["outcome"] == "success"
        assert "pytest.failed" not in result["markers_matched"]

    def test_jest_zero_failed_summary_is_success(self):
        result = derive_outcome(
            rc=0, is_error=False, interrupted=False,
            stdout_digest="Tests:       0 failed, 5 passed, 5 total", stderr_digest="",
        )
        assert result["outcome"] == "success"
        assert "jest.failed" not in result["markers_matched"]

    def test_green_cargo_summary_with_zero_failed_is_success_not_soft_failure(self):
        """The reviewer's specific witness case: cargo's own summary line
        contains the substring "0 failed", and prior to this fix the
        generic (non-cargo-scoped) `pytest.failed` pattern matched it
        anywhere in the digest, downgrading a green run (rc=0) to
        soft_failure."""
        result = derive_outcome(
            rc=0, is_error=False, interrupted=False,
            stdout_digest=(
                "running 5 tests\n"
                "test result: ok. 5 passed; 0 failed; 0 ignored; "
                "0 measured; 0 filtered out"
            ),
            stderr_digest="",
        )
        assert result["outcome"] == "success"
        assert "pytest.failed" not in result["markers_matched"]

    def test_nonzero_failed_count_still_matches(self):
        """The fix narrows the pattern to [1-9]\\d* — a genuine failure
        count must still be detected."""
        result = derive_outcome(
            rc=1, is_error=False, interrupted=False,
            stdout_digest="4 passed, 2 failed in 1.2s", stderr_digest="",
        )
        assert result["outcome"] == "failure"
        assert "pytest.failed" in result["markers_matched"]

    def test_jest_nonzero_failed_count_still_matches(self):
        result = derive_outcome(
            rc=1, is_error=False, interrupted=False,
            stdout_digest="Tests:       2 failed, 3 passed, 5 total", stderr_digest="",
        )
        assert result["outcome"] == "failure"
        assert "jest.failed" in result["markers_matched"]


class TestInterruptedWinsOverRcAndMarkers:
    """Adversarial re-review fix (2026-07-22, spec trap (d)): a
    user-interrupted command (Ctrl-C -> SIGINT, or a SIGTERM) must derive
    outcome="interrupted" — never "failure"/"soft_failure" — regardless of
    what its exit code or printed output look like. Before this fix,
    `elif rc is not None` ran before any interrupted check, so
    PostToolUseFailure's rc=130/143 (128+signum, POSIX convention) with
    is_interrupt=True fell straight into the rc!=0 -> "failure" branch,
    and a Ctrl-C mid-test that happened to print a partial "N failed"
    summary line hit the marker-failure branch first regardless. Both
    must be inert once `interrupted` is set — the run never completed, so
    it can never become an arc endpoint (app/arcs.py's pending guard)."""

    def test_sigint_rc_130_with_interrupted_flag_is_interrupted(self):
        result = derive_outcome(
            rc=130, is_error=True, interrupted=True,
            stdout_digest="", stderr_digest="Exit code 130",
        )
        assert result["outcome"] == "interrupted"
        assert result["termination"] == "cancelled"

    def test_sigterm_rc_143_with_interrupted_flag_is_interrupted(self):
        result = derive_outcome(
            rc=143, is_error=True, interrupted=True,
            stdout_digest="", stderr_digest="Exit code 143",
        )
        assert result["outcome"] == "interrupted"
        assert result["termination"] == "cancelled"

    def test_sigterm_rc_143_without_interrupted_flag_is_still_interrupted(self):
        """A raw 128+signum exit code alone (no editor-supplied `interrupted`
        flag) already meant termination="signal" pre-fix; must still map to
        outcome="interrupted", not "failure"."""
        result = derive_outcome(
            rc=143, is_error=False, interrupted=False,
            stdout_digest="", stderr_digest="",
        )
        assert result["outcome"] == "interrupted"
        assert result["termination"] == "signal"

    def test_failure_marker_in_interrupted_output_does_not_override(self):
        """Reviewer's precedence case: a Ctrl-C mid-`pytest` run can still
        print a "N failed" partial summary before the process dies — that
        marker match must not win over `interrupted`."""
        result = derive_outcome(
            rc=130, is_error=True, interrupted=True,
            stdout_digest="", stderr_digest="2 failed, 1 passed in 0.4s",
        )
        assert result["outcome"] == "interrupted"
        assert "pytest.failed" in result["markers_matched"]  # matched, but not decisive

    def test_success_marker_in_interrupted_output_does_not_override(self):
        result = derive_outcome(
            rc=130, is_error=True, interrupted=True,
            stdout_digest="5 passed in 0.4s", stderr_digest="",
        )
        assert result["outcome"] == "interrupted"
