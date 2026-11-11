"""Tests for app/arcs.py — deterministic streaming failure->success arc
detection (L1 capture design doc §3, LANE-ARC).

Table-driven: similarity scoring (§3.2), the pending-window state machine
(§3.3: chains, interleaving, TTL aging), edit-mediated vs. flaky
disambiguation (§3.4), and every false-positive trap (§3.5).

Every episode here models a TOOL-CALL record (argv structure / exit
outcome of an already-executed Bash/Edit call) — never prompt/query
content — matching the R5 scope of app/arcs.py itself.
"""
from __future__ import annotations

from typing import Any

from app.arcs import ArcDetector, is_identical_command, is_mutation_band, similarity
from app.cmdnorm import normalize_command as nc


def make_episode(
    session_id: str = "s1",
    ts: float = 0.0,
    cwd: str = "/repo",
    tool: str = "bash",
    cmd_raw: str = "",
    outcome: str | None = None,
    termination: str | None = None,
    markers: list[str] | None = None,
    env_delta_names: list[str] | None = None,
    file_path: str | None = None,
) -> dict[str, Any]:
    return dict(
        session_id=session_id,
        ts=ts,
        cwd=cwd,
        tool=tool,
        cmd_raw=cmd_raw,
        outcome=outcome,
        termination=termination,
        markers=markers or [],
        env_delta_names=env_delta_names or [],
        file_path=file_path,
    )


# ---------------------------------------------------------------------------
# Similarity (§3.2)
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_identical_command_scores_one(self) -> None:
        a = nc("git status")
        b = nc("git status")
        assert similarity(a, b) == 1.0
        assert is_identical_command(a, b)
        assert not is_mutation_band(similarity(a, b)), "identical (1.0) is never a mutation"

    def test_flag_value_change_is_mutation(self) -> None:
        a = nc("mvn test -Dtest=Foo")
        b = nc("mvn test -Dtest=Bar")
        score = similarity(a, b)
        assert is_mutation_band(score)
        assert not is_identical_command(a, b)

    def test_verb_typo_soft_match(self) -> None:
        # A single-character verb typo still passes the Levenshtein soft-
        # match ratio and contributes partial (not full) verb score.
        a = nc("gt status")
        b = nc("git status")
        score = similarity(a, b)
        assert score > 0.0
        assert score < 1.0

    def test_unrelated_commands_score_below_band(self) -> None:
        a = nc("git status")
        b = nc("ls -la /tmp")
        assert similarity(a, b) < 0.55
        assert not is_mutation_band(similarity(a, b))

    def test_unrelated_install_targets_not_a_mutation(self) -> None:
        # Different, unrelated package targets under the same tool/verb
        # must NOT look like a mutation of each other.
        a = nc("pip install flask")
        b = nc("pip install django")
        assert not is_mutation_band(similarity(a, b))

    def test_path_swap_same_arity_uses_positional_levenshtein(self) -> None:
        a = nc("cat src/foo.py")
        b = nc("cat src/bar.py")
        score = similarity(a, b)
        assert is_mutation_band(score)

    def test_flag_jaccard_partial_overlap(self) -> None:
        a = nc("ls -l -a")
        b = nc("ls -l -h")
        score = similarity(a, b)
        # verb=1.0*0.5=0.5, flags jaccard({-a,-l} vs {-h,-l})=1/3*0.3=0.1,
        # args both empty -> 1.0*0.2=0.2; total=0.8
        assert abs(score - 0.8) < 1e-9

    def test_different_arity_args_falls_back_to_class_jaccard(self) -> None:
        a = nc("cp src/a.py src/b.py")
        b = nc("cp src/a.py")
        score = similarity(a, b)
        # arity differs (2 vs 1) -> jaccard over arg_classes, both all
        # <PATH> -> jaccard = 1.0 despite different arities.
        assert is_mutation_band(score)


# ---------------------------------------------------------------------------
# Detector state machine — basic mutation arcs (§3.3)
# ---------------------------------------------------------------------------


class TestBasicMutationArc:
    def test_single_failure_then_mutated_success_emits_arc(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="mvn test -Dtest=Foo", outcome="failure"))
        arcs = d.observe(make_episode(ts=1, cmd_raw="mvn test -Dtest=Bar", outcome="success"))
        assert len(arcs) == 1
        arc = arcs[0]
        assert arc.mutation_diff == {"flag": (("-Dtest=Foo",), ("-Dtest=Bar",))}
        assert arc.confidence == "normal"
        assert len(arc.failures_chain) == 1

    def test_matched_failure_is_consumed_from_pending(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="mvn test -Dtest=Foo", outcome="failure"))
        d.observe(make_episode(ts=1, cmd_raw="mvn test -Dtest=Bar", outcome="success"))
        state = d._sessions["s1"]
        assert state.pending == {}

    def test_success_with_no_pending_failure_emits_nothing(self) -> None:
        d = ArcDetector()
        arcs = d.observe(make_episode(ts=0, cmd_raw="git status", outcome="success"))
        assert arcs == []

    def test_unrelated_failure_left_unresolved_by_unrelated_success(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pip install flask", outcome="failure"))
        arcs = d.observe(make_episode(ts=1, cmd_raw="pip install django", outcome="success"))
        assert arcs == []
        state = d._sessions["s1"]
        assert len(state.pending[("pip", "/repo")]) == 1


# ---------------------------------------------------------------------------
# Chains and interleaving (§3.3)
# ---------------------------------------------------------------------------


class TestChainsAndInterleaving:
    def test_chain_backward_through_pending_failures(self) -> None:
        # install -> missing dep -> pin version -> succeeds (§3.3 example).
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pip install requests", outcome="failure"))
        d.observe(make_episode(ts=1, cmd_raw="pip install requests==2.28.0", outcome="failure"))
        arcs = d.observe(make_episode(ts=2, cmd_raw="pip install requests==2.28.1", outcome="success"))
        assert len(arcs) == 1
        arc = arcs[0]
        assert [f["cmd_raw"] for f in arc.failures_chain] == [
            "pip install requests",
            "pip install requests==2.28.0",
        ]

    def test_interleaved_unrelated_failure_is_skipped_not_a_stop_condition(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pip install requests", outcome="failure"))
        d.observe(
            make_episode(
                ts=1,
                cmd_raw="pip install completely-different-package-xyz-not-related-at-all",
                outcome="failure",
            )
        )
        d.observe(make_episode(ts=2, cmd_raw="pip install requests==2.28.0", outcome="failure"))
        arcs = d.observe(make_episode(ts=3, cmd_raw="pip install requests==2.28.1", outcome="success"))
        assert len(arcs) == 1
        chain_cmds = [f["cmd_raw"] for f in arcs[0].failures_chain]
        assert chain_cmds == ["pip install requests", "pip install requests==2.28.0"]
        assert "pip install completely-different-package-xyz-not-related-at-all" not in chain_cmds

    def test_interleaved_unrelated_failure_remains_pending_afterward(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pip install requests", outcome="failure"))
        d.observe(
            make_episode(
                ts=1,
                cmd_raw="pip install completely-different-package-xyz-not-related-at-all",
                outcome="failure",
            )
        )
        d.observe(make_episode(ts=2, cmd_raw="pip install requests==2.28.0", outcome="failure"))
        d.observe(make_episode(ts=3, cmd_raw="pip install requests==2.28.1", outcome="success"))
        state = d._sessions["s1"]
        remaining = [p.episode["cmd_raw"] for p in state.pending.get(("pip", "/repo"), [])]
        assert remaining == ["pip install completely-different-package-xyz-not-related-at-all"]

    def test_interleaved_different_family_success_is_ignored(self) -> None:
        # A totally different tool's success in between must not touch or
        # consume the still-pending failure.
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="mvn test -Dtest=Foo", outcome="failure"))
        d.observe(make_episode(ts=1, cmd_raw="git status", outcome="success"))
        arcs = d.observe(make_episode(ts=2, cmd_raw="mvn test -Dtest=Bar", outcome="success"))
        assert len(arcs) == 1

    def test_tie_break_prefers_most_recent_equally_similar_failure(self) -> None:
        # Two pending failures score identically against the success; the
        # more recent one must be picked as the anchor so the chain-walk
        # still reaches back through the older one, rather than stranding
        # it unconsumed while an equally-scored but older failure "wins".
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="git checkout feature-a", outcome="failure"))
        d.observe(make_episode(ts=1, cmd_raw="git checkout feature-b", outcome="failure"))
        arcs = d.observe(make_episode(ts=2, cmd_raw="git checkout feature-c", outcome="success"))
        assert len(arcs) == 1
        state = d._sessions["s1"]
        assert state.pending == {}


# ---------------------------------------------------------------------------
# TTL / command-count aging (§3.3)
# ---------------------------------------------------------------------------


class TestAging:
    def test_pending_failure_ages_out_by_ttl(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pytest tests/test_foo.py", outcome="failure"))
        arcs = d.observe(make_episode(ts=700, cmd_raw="pytest tests/test_foo.py", outcome="success"))
        assert arcs == []

    def test_pending_failure_survives_within_ttl(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pytest tests/test_foo.py", outcome="failure"))
        arcs = d.observe(make_episode(ts=599, cmd_raw="pytest tests/test_foo.py -v", outcome="success"))
        assert len(arcs) == 1

    def test_pending_failure_ages_out_by_command_count(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pytest tests/test_foo.py", outcome="failure"))
        for i in range(1, 22):
            d.observe(make_episode(ts=float(i), cmd_raw=f"echo noop{i}", outcome="success"))
        arcs = d.observe(make_episode(ts=25.0, cmd_raw="pytest tests/test_foo.py", outcome="success"))
        assert arcs == []

    def test_pending_failure_survives_within_command_count(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pytest tests/test_foo.py", outcome="failure"))
        for i in range(1, 19):
            d.observe(make_episode(ts=float(i), cmd_raw=f"echo noop{i}", outcome="success"))
        arcs = d.observe(make_episode(ts=20.0, cmd_raw="pytest tests/test_foo.py -v", outcome="success"))
        assert len(arcs) == 1


# ---------------------------------------------------------------------------
# Edit-mediated vs. flaky retry (§3.4)
# ---------------------------------------------------------------------------


class TestEditMediatedVsFlaky:
    def test_intervening_edit_produces_edit_mediated_arc(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pytest tests/test_foo.py", outcome="failure"))
        d.observe(make_episode(ts=1, tool="edit", file_path="app/foo.py"))
        arcs = d.observe(make_episode(ts=2, cmd_raw="pytest tests/test_foo.py", outcome="success"))
        assert len(arcs) == 1
        assert arcs[0].mutation_diff == {"files": ((), ("app/foo.py",))}

    def test_multiple_intervening_edits_collected(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pytest tests/test_foo.py", outcome="failure"))
        d.observe(make_episode(ts=1, tool="edit", file_path="app/foo.py"))
        d.observe(make_episode(ts=2, tool="edit", file_path="app/bar.py"))
        arcs = d.observe(make_episode(ts=3, cmd_raw="pytest tests/test_foo.py", outcome="success"))
        assert len(arcs) == 1
        assert arcs[0].mutation_diff["files"] == ((), ("app/bar.py", "app/foo.py"))

    def test_identical_retry_no_edit_no_delta_is_flaky_suppressed(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="curl https://api.example.com/health", outcome="failure"))
        arcs = d.observe(make_episode(ts=1, cmd_raw="curl https://api.example.com/health", outcome="success"))
        assert arcs == []
        state = d._sessions["s1"]
        sig = ("curl", frozenset(), ("https://api.example.com/health",))
        assert state.flake_counts[sig] == 1
        assert state.pending == {}

    def test_cross_cwd_identical_command_never_matches(self) -> None:
        """§3.3 (review 2026-07-22, reviewer finding F1/G1): cwd is a
        pending-bucket key, NOT a mutation axis — a failure in one cwd and
        an identical-command success in a DIFFERENT cwd must never be
        bound together (they land in different buckets entirely), or
        unrelated repos' builds would falsely bind into an arc."""
        d = ArcDetector()
        d.observe(make_episode(ts=0, cwd="/work/serviceA", cmd_raw="make build", outcome="failure"))
        arcs = d.observe(make_episode(ts=1, cwd="/work/serviceB", cmd_raw="make build", outcome="success"))
        assert arcs == []
        # The serviceA failure is untouched, sitting in its own bucket.
        state = d._sessions["s1"]
        assert len(state.pending[("make", "/work/serviceA")]) == 1

    def test_cross_cwd_mutation_candidate_never_matches(self) -> None:
        """§3.3 (review 2026-07-22, reviewer finding F1/G2): the same
        cross-cwd isolation applies to the mutation-band (non-identical)
        path, not just the identical-command path."""
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/work/serviceA", cmd_raw="mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/work/serviceB", cmd_raw="mvn test -Dtest=Bar", outcome="success")
        )
        assert arcs == []

    def test_leading_cd_prefix_different_targets_never_matches(self) -> None:
        """UPG-ARC-CWD-VS-EFFECTIVE-DIR: the episode's `cwd` FIELD is only
        where the harness started the command, not necessarily where it
        actually ran — an agent that writes `cd /other/repo && cmd` inside
        `cmd_raw` changes the EFFECTIVE directory without changing the
        episode's own cwd field. A failure and a success recorded under the
        SAME episode cwd but with DIFFERENT leading `cd` targets in
        cmd_raw must land in different buckets, exactly like
        test_cross_cwd_mutation_candidate_never_matches above but with the
        directory divergence hidden inside cmd_raw instead of the cwd
        field."""
        d = ArcDetector()
        d.observe(
            make_episode(
                ts=0, cwd="/harness", cmd_raw="cd /repo-a && mvn test -Dtest=Foo", outcome="failure"
            )
        )
        arcs = d.observe(
            make_episode(
                ts=1, cwd="/harness", cmd_raw="cd /repo-b && mvn test -Dtest=Bar", outcome="success"
            )
        )
        assert arcs == []

    def test_leading_cd_prefix_same_target_still_matches(self) -> None:
        """Companion to the test above: when the leading `cd` target is the
        SAME across the failure and the success (the common real-world
        case — an agent that always `cd`s into the repo before running a
        command), the mutation-band match must still fire exactly as it
        does with no `cd` prefix at all."""
        d = ArcDetector()
        d.observe(
            make_episode(
                ts=0, cwd="/harness", cmd_raw="cd /repo-a && mvn test -Dtest=Foo", outcome="failure"
            )
        )
        arcs = d.observe(
            make_episode(
                ts=1, cwd="/harness", cmd_raw="cd /repo-a && mvn test -Dtest=Bar", outcome="success"
            )
        )
        assert len(arcs) == 1
        assert arcs[0].mutation_diff == {"flag": (("-Dtest=Foo",), ("-Dtest=Bar",))}

    def test_identical_retry_with_env_delta_is_not_flaky(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="make build", outcome="failure", env_delta_names=[]))
        arcs = d.observe(
            make_episode(ts=1, cmd_raw="make build", outcome="success", env_delta_names=["CC"])
        )
        assert len(arcs) == 1
        assert arcs[0].mutation_diff == {"env": ((), ("CC",))}

    def test_near_threshold_match_suppressed_once_proven_flaky(self) -> None:
        d = ArcDetector()
        cmd = "curl https://api.example.com/health"
        # Two identical flaky retries prove the command flaky.
        d.observe(make_episode(ts=0, cmd_raw=cmd, outcome="failure"))
        d.observe(make_episode(ts=1, cmd_raw=cmd, outcome="success"))
        d.observe(make_episode(ts=2, cmd_raw=cmd, outcome="failure"))
        d.observe(make_episode(ts=3, cmd_raw=cmd, outcome="success"))
        state = d._sessions["s1"]
        sig = ("curl", frozenset(), ("https://api.example.com/health",))
        assert state.flake_counts[sig] == 2

        # A near-identical (trailing slash) success now arrives after
        # another identical fail — must be suppressed as flaky noise, not
        # emitted as a real mutation arc.
        d.observe(make_episode(ts=4, cmd_raw=cmd, outcome="failure"))
        arcs = d.observe(make_episode(ts=5, cmd_raw=cmd + "/", outcome="success"))
        assert arcs == []
        assert state.pending == {}

    def test_near_threshold_match_not_suppressed_before_flaky_is_proven(self) -> None:
        d = ArcDetector()
        cmd = "curl https://api.example.com/health"
        # Only ONE flaky retry so far — below suppress_min_count (2).
        d.observe(make_episode(ts=0, cmd_raw=cmd, outcome="failure"))
        d.observe(make_episode(ts=1, cmd_raw=cmd, outcome="success"))
        d.observe(make_episode(ts=2, cmd_raw=cmd, outcome="failure"))
        arcs = d.observe(make_episode(ts=3, cmd_raw=cmd + "/", outcome="success"))
        assert len(arcs) == 1
        assert arcs[0].mutation_diff == {
            "arg": (
                ("https://api.example.com/health",),
                ("https://api.example.com/health/",),
            )
        }


# ---------------------------------------------------------------------------
# Effective-cwd bucketing: relative composition + refusal
# (UPG-ARC-CWD-VS-EFFECTIVE-DIR refinement of commit 7cbd847)
# ---------------------------------------------------------------------------


class TestEffectiveCwdBucketing:
    """7cbd847 already bucketed on the RAW leading-cd target text; this
    refinement composes RELATIVE targets with the episode's cwd field,
    keys tilde/$HOME forms symbolically, and REFUSES TO PAIR entirely on
    targets no static string can resolve (`cd -`, `$var`, declined shapes).
    Refusal bias: an arc that is not offered costs a missed lesson; an arc
    offered under a guessed directory can persist a false lesson into
    memory permanently."""

    def test_relative_cd_composes_with_cwd_field_across_repos(self) -> None:
        # HEADLINE: both episodes share NO directory but their textual cd
        # target agrees — with the shipped textual key both landed in
        # bucket ("mvn", "sub") and the /work/A failure paired with the
        # /work/B success (a lesson about a repo pairing that never
        # happened). Composed keys /work/A/sub and /work/B/sub differ.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/work/A", cmd_raw="cd sub && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/work/B", cmd_raw="cd sub && mvn test -Dtest=Bar", outcome="success")
        )
        assert arcs == []
        state = d._sessions["s1"]
        assert len(state.pending[("mvn", "/work/A/sub")]) == 1

    def test_relative_cd_same_base_still_pairs(self) -> None:
        # Composition must not kill the genuine same-repo case.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/work/A", cmd_raw="cd sub && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/work/A", cmd_raw="cd sub && mvn test -Dtest=Bar", outcome="success")
        )
        assert len(arcs) == 1
        assert arcs[0].mutation_diff == {"flag": (("-Dtest=Foo",), ("-Dtest=Bar",))}

    def test_absolute_cd_replaces_harness_cwd(self) -> None:
        # When the command cds to an absolute path the harness cwd fields
        # are irrelevant — both episodes really ran in /shared/repo, so
        # they pair even though every other cwd signal disagrees.
        d = ArcDetector()
        d.observe(
            make_episode(
                ts=0, cwd="/harness/first", cmd_raw="cd /shared/repo && mvn test -Dtest=Foo", outcome="failure"
            )
        )
        arcs = d.observe(
            make_episode(
                ts=1, cwd="/harness/second", cmd_raw="cd /shared/repo && mvn test -Dtest=Bar", outcome="success"
            )
        )
        assert len(arcs) == 1

    def test_bare_cd_keys_symbolic_home_not_cwd_field(self) -> None:
        # `cd && mvn ...` runs in $HOME; plain `mvn ...` runs in the
        # harness cwd. Same cwd FIELD, different effective directories ->
        # never pair. With the shipped fallback (bare cd -> cwd field)
        # both keyed identically and WOULD have paired.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/w/A", cmd_raw="cd && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/w/A", cmd_raw="mvn test -Dtest=Bar", outcome="success")
        )
        assert arcs == []
        state = d._sessions["s1"]
        assert len(state.pending[("mvn", "~")]) == 1

    def test_two_bare_cd_commands_pair_as_same_home(self) -> None:
        # Both episodes genuinely ran in $HOME regardless of their harness
        # cwd fields, so the symbolic key groups them truthfully.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/w/A", cmd_raw="cd && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/w/B", cmd_raw="cd && mvn test -Dtest=Bar", outcome="success")
        )
        assert len(arcs) == 1

    def test_tilde_target_pairs_across_different_cwd_fields(self) -> None:
        # `~/repo` resolves to the SAME $HOME-derived directory on this
        # machine for both episodes; the tilde-form is the truthful key.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/w/A", cmd_raw="cd ~/repo && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/w/B", cmd_raw="cd ~/repo && mvn test -Dtest=Bar", outcome="success")
        )
        assert len(arcs) == 1

    def test_oldpwd_dash_refuses_pairing_entirely(self) -> None:
        # `cd -` means $OLDPWD — unknowable from a static string and
        # different at every call site. Neither episode may enter pending
        # nor resolve one; with any shared-key guess these two would pair.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/w/A", cmd_raw="cd - && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/w/A", cmd_raw="cd - && mvn test -Dtest=Bar", outcome="success")
        )
        assert arcs == []
        assert d._sessions["s1"].pending == {}

    def test_env_var_target_refuses_pairing_entirely(self) -> None:
        # $REPO's VALUE can change between call sites even though the NAME
        # recurs — a textual key would falsely group two different repos,
        # so refusal is the only truthful verdict.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/w/A", cmd_raw="cd $REPO && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/w/A", cmd_raw="cd $REPO && mvn test -Dtest=Bar", outcome="success")
        )
        assert arcs == []
        assert d._sessions["s1"].pending == {}

    def test_flagged_cd_declined_shape_refuses_pairing(self) -> None:
        # BEHAVIOR CHANGE vs 7cbd847, which fell back to the cwd field for
        # shapes it declined parsing (flagged `cd -- <path>`): such a
        # command PROVABLY did not run in the harness cwd, so trusting that
        # field is knowingly wrong. Refusal is the honest verdict.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/w/A", cmd_raw="cd -- /repo-a && mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/w/A", cmd_raw="cd -- /repo-a && mvn test -Dtest=Bar", outcome="success")
        )
        assert arcs == []
        assert d._sessions["s1"].pending == {}

    def test_glued_separator_cd_composes_too(self) -> None:
        # Items (1)+(2) interplay: the glued-separator tokenizer fix feeds
        # the classifier, so even `cd sub;mvn ...` composes per-harness-cwd.
        d = ArcDetector()
        d.observe(
            make_episode(ts=0, cwd="/work/A", cmd_raw="cd sub;mvn test -Dtest=Foo", outcome="failure")
        )
        arcs = d.observe(
            make_episode(ts=1, cwd="/work/B", cmd_raw="cd sub;mvn test -Dtest=Bar", outcome="success")
        )
        assert arcs == []


# ---------------------------------------------------------------------------
# False-positive traps (§3.5)
# ---------------------------------------------------------------------------


class TestTraps:
    def test_transient_marker_tags_low_confidence(self) -> None:
        d = ArcDetector()
        d.observe(
            make_episode(
                ts=0,
                cmd_raw="curl https://api.example.com/data",
                outcome="failure",
                markers=["generic.timeout"],
            )
        )
        arcs = d.observe(
            make_episode(ts=1, cmd_raw="curl https://api.example.com/data --retry 3", outcome="success")
        )
        assert len(arcs) == 1
        assert arcs[0].confidence == "low"

    def test_no_transient_marker_is_normal_confidence(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="mvn test -Dtest=Foo", outcome="failure", markers=["maven.build_failure"]))
        arcs = d.observe(make_episode(ts=1, cmd_raw="mvn test -Dtest=Bar", outcome="success"))
        assert len(arcs) == 1
        assert arcs[0].confidence == "normal"

    def test_interrupted_never_enters_pending(self) -> None:
        d = ArcDetector()
        d.observe(
            make_episode(
                ts=0, cmd_raw="pytest tests/test_foo.py", outcome="interrupted", termination="interrupted"
            )
        )
        state = d._sessions["s1"]
        assert state.pending == {}
        arcs = d.observe(make_episode(ts=1, cmd_raw="pytest tests/test_foo.py", outcome="success"))
        assert arcs == []

    def test_unknown_outcome_never_enters_pending(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="pytest tests/test_foo.py", outcome="unknown"))
        state = d._sessions["s1"]
        assert state.pending == {}

    def test_soft_failure_enters_pending_like_failure(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="mvn test -Dtest=Foo", outcome="soft_failure"))
        arcs = d.observe(make_episode(ts=1, cmd_raw="mvn test -Dtest=Bar", outcome="success"))
        assert len(arcs) == 1

    def test_interrupted_with_real_termination_value_never_enters_pending(self) -> None:
        """Adversarial re-review (2026-07-22): `termination` is never
        literally "interrupted" (agent/outcome.py's TERMINATION_VALUES are
        normal/signal/timeout/cancelled/unknown) — the old
        `episode.get("termination") == "interrupted"` guard was dead code.
        The real values a Ctrl-C (SIGINT, rc=130) or SIGTERM (rc=143)
        produce are outcome="interrupted" with termination="cancelled" or
        "signal". Both must be independently sufficient to keep the episode
        out of pending, per the broadened _INTERRUPTED_TERMINATIONS guard."""
        for outcome, termination in (("interrupted", "cancelled"), ("interrupted", "signal")):
            d = ArcDetector()
            d.observe(
                make_episode(
                    ts=0, cmd_raw="pytest tests/x.py", outcome=outcome, termination=termination
                )
            )
            state = d._sessions["s1"]
            assert state.pending == {}
            arcs = d.observe(make_episode(ts=1, cmd_raw="pytest tests/x.py", outcome="success"))
            assert arcs == []

    def test_interrupted_edit_then_identical_success_emits_zero_arcs(self) -> None:
        """Reviewer's exact end-to-end repro: Ctrl-C'd `pytest tests/x.py`
        -> edit x.py -> identical green rerun must never be read as a
        resolved failure->success arc — the "failure" never happened, the
        run was cancelled."""
        d = ArcDetector()
        d.observe(
            make_episode(
                ts=0, cmd_raw="pytest tests/x.py", outcome="interrupted", termination="cancelled"
            )
        )
        d.observe(make_episode(ts=1, tool="edit", file_path="/repo/tests/x.py"))
        arcs = d.observe(make_episode(ts=2, cmd_raw="pytest tests/x.py", outcome="success"))
        assert arcs == []


# ---------------------------------------------------------------------------
# Episode/session plumbing
# ---------------------------------------------------------------------------


class TestEpisodePlumbing:
    def test_missing_session_id_ignored(self) -> None:
        d = ArcDetector()
        arcs = d.observe(make_episode(session_id="", ts=0, cmd_raw="git status", outcome="failure"))
        assert arcs == []

    def test_non_bash_non_edit_tool_ignored(self) -> None:
        d = ArcDetector()
        arcs = d.observe(make_episode(ts=0, tool="read", cmd_raw="git status", outcome="failure"))
        assert arcs == []
        assert d._sessions == {}

    def test_sessions_are_isolated(self) -> None:
        d = ArcDetector()
        d.observe(make_episode(session_id="s1", ts=0, cmd_raw="mvn test -Dtest=Foo", outcome="failure"))
        arcs = d.observe(make_episode(session_id="s2", ts=1, cmd_raw="mvn test -Dtest=Bar", outcome="success"))
        assert arcs == [], "a different session must never resolve another session's pending failure"

    def test_edit_episode_never_emits_an_arc_itself(self) -> None:
        d = ArcDetector()
        arcs = d.observe(make_episode(ts=0, tool="edit", file_path="app/foo.py"))
        assert arcs == []


# ---------------------------------------------------------------------------
# Reviewer findings F2-F5 (2026-07-22 adversarial review of bcbcb50):
# pipeline collapse, wrapper transparency, empty-verb handling, ts
# robustness. F1 (cwd bucket key) is covered above in
# TestEditMediatedVsFlaky.test_cross_cwd_identical_command_never_matches /
# test_cross_cwd_mutation_candidate_never_matches.
# ---------------------------------------------------------------------------


class TestPipelineWrapperEmptyVerbAndTsRobustness:
    def test_multistage_pipeline_edit_then_different_downstream_stage_is_not_edit_mediated(
        self,
    ) -> None:
        """F2: `cmdnorm` must keep a non-trailing downstream pipeline
        stage's tokens in the comparison set, so two genuinely different
        pipelines are never treated as the identical command an
        intervening edit could "resolve"."""
        d = ArcDetector()
        assert not is_identical_command(
            nc("cat data.csv | python train.py"), nc("cat data.csv | python eval.py")
        )
        d.observe(make_episode(ts=0, cmd_raw="cat data.csv | python train.py", outcome="failure"))
        d.observe(make_episode(ts=1, tool="edit", file_path="unrelated.txt"))
        arcs = d.observe(make_episode(ts=2, cmd_raw="cat data.csv | python eval.py", outcome="success"))
        for arc in arcs:
            assert "files" not in arc.mutation_diff, (
                "a genuinely different downstream pipeline stage must never be classified "
                "edit-mediated (that branch is reserved for an IDENTICAL command)"
            )

    def test_wrapper_prefix_timeout_value_change_does_not_dodge_flaky_suppression(self) -> None:
        """F3: `timeout N` is a transparent wrapper — the wrapped command
        is the verb, so two retries differing only in the timeout value
        must normalize IDENTICALLY and go through the flaky-retry path,
        not be scored as a distinct mutation."""
        assert is_identical_command(
            nc("timeout 60 curl https://api.example.com/x"),
            nc("timeout 90 curl https://api.example.com/x"),
        )
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="timeout 60 curl https://api.example.com/x", outcome="failure"))
        arcs = d.observe(
            make_episode(ts=1, cmd_raw="timeout 90 curl https://api.example.com/x", outcome="success")
        )
        assert arcs == [], "no intervening edit, no env delta -> flaky retry, suppressed"
        state = d._sessions["s1"]
        assert state.flake_counts

    def test_empty_verb_episode_never_enters_or_resolves_pending(self) -> None:
        """F4: an episode that normalizes to an empty verb (`2>&1` alone,
        an env-assignment-only command) is treated like outcome `unknown`
        — never enters pending, never resolves a pending failure."""
        d = ArcDetector()
        d.observe(make_episode(ts=0, cmd_raw="2>&1", outcome="failure"))
        d.observe(make_episode(ts=1, tool="edit", file_path="app/foo.py"))
        arcs = d.observe(make_episode(ts=2, cmd_raw="FOO=bar", outcome="success"))
        assert arcs == []
        state = d._sessions["s1"]
        assert state.pending == {}

    def test_observe_tolerates_missing_ts_key(self) -> None:
        """F5: observe() must never raise on a missing `ts` field — the
        wiring lane may not always wrap every call."""
        d = ArcDetector()
        episode = make_episode(ts=0, cmd_raw="git status", outcome="failure")
        del episode["ts"]
        arcs = d.observe(episode)
        assert arcs == []

    def test_observe_tolerates_none_ts_value(self) -> None:
        """F5: same tolerance for an explicit `ts=None` (a drift-tolerant
        payload that supplied the key but not a value) — the monotonic
        fallback must still let a genuine mutation match resolve to an
        arc rather than raising or silently losing the pending failure."""
        d = ArcDetector()
        d.observe(make_episode(ts=None, cmd_raw="mvn test -Dtest=Foo", outcome="failure"))
        arcs = d.observe(make_episode(ts=None, cmd_raw="mvn test -Dtest=Bar", outcome="success"))
        assert len(arcs) == 1
