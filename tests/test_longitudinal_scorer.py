"""Discrimination and invariant tests for the longitudinal-rediscovery mechanical scorer.

The point of the discrimination tests is anti-vacuity, exactly as in
`tests/test_injection_utility_scorer.py`: a scorer that returns the SAME verdict for
an agent that re-derives a fact the hard way (and along the way commits the scenario's
documented mistake) and one that uses the fact directly would make every longitudinal
A/B comparison a guaranteed tie, while looking well-behaved. For every scenario this
file builds two hand-written fixtures against a freshly materialized workspace -- one
"naive" (commits the mistake, never reaches the fact) and one "fact-using" (acquires
the fact directly, avoids the mistake, and satisfies the leg's real checks) -- and
asserts `score_run`/`leg_metrics` separate them on the primary check AND on
`turns_to_fact` (DESIGN.md section 11.6).

The remaining tests in this file cover the other section 11.6 requirements directly at
the primitive level: censoring never imputes from session totals; a missing transcript
degrades to "not acquired" rather than raising; `leg_non_vacuity`'s anchor matching is
exact (`note:4` is never satisfied by `note:41`); `score_run`/`leg_metrics` take no
`arm` argument; and the cross-scenario structural invariants scenarios.py's own
`__post_init__` and docstrings promise (fact-token confinement, the forcing-leg flag,
verify scripts never landing inside a workspace, every note variant carrying the fact
sentence byte-identically).

`mistake_repetition_rate is null when leg 1 is clean` is NOT tested here: it is a
TRAJECTORY-level aggregate across sibling legs that `leg_metrics()`'s own docstring
says explicitly belongs to `report.py` (not yet implemented -- see the coder-lane
report for this task), not to this per-leg scorer. What IS tested here is the leg-level
primitive that aggregate will read: `weak_prior` (`test_weak_prior_flags_a_clean_leg1_on_a_discovered_scenario`).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "longitudinal_rediscovery"

# `benchmarks/injection_utility/` ships its own same-named `scenarios.py`/`scorer.py`,
# and `tests/test_injection_utility_scorer.py` imports them under those bare names. A
# bare `sys.path.insert` + `import scenarios`/`import scorer` here would race that
# file for `sys.modules["scenarios"]`/`["scorer"]` -- whichever test module pytest
# collects first wins, and the loser silently binds the wrong harness's module
# (observed: `scen.get()` resolving to the trap harness's slugs). Loading by explicit
# file path under a fixed, cache-checked key sidesteps the collision regardless of
# collection order. `scenarios.py` uses the identical `_vectr_eval_longitudinal_scenarios`
# key when `scorer.py` loads it internally, so this file's `scen` and `scorer`'s own
# check-primitive classes always converge on ONE module object.
_LONGITUDINAL_SCENARIOS_KEY = "_vectr_eval_longitudinal_scenarios"


def _load_local_module(key: str, filename: str):
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, BENCH_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scen = _load_local_module(_LONGITUDINAL_SCENARIOS_KEY, "scenarios.py")
scorer = _load_local_module("_vectr_eval_longitudinal_scorer_test", "scorer.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _transcript(
    actions: list[tuple[str, dict[str, Any]]],
    *,
    mcp_servers: list[dict[str, str]] | None = None,
    tools: list[str] | None = None,
    num_turns: int = 4,
    usage: dict[str, Any] | None = None,
) -> list[dict]:
    """A minimal stream-json transcript carrying the given ordered tool calls.

    One `assistant` event per action (not batched) so each action's `event_index`
    lands on its own event, matching how `leg_metrics` reads the acquiring event's
    own usage for `context_tokens_at_fact`. Each event gets a synthetic, per-event
    unique `message.id` -- `_dedupe_assistant_messages` (UPG-EVAL-DEDUPE-NO-
    MESSAGE-ID) raises on a missing id, and every real event carries one, so a
    fixture with none would not represent a real transcript shape.
    """
    usage = usage or {"input_tokens": 100, "output_tokens": 50}
    events: list[dict] = [
        {
            "type": "system",
            "subtype": "init",
            "mcp_servers": mcp_servers if mcp_servers is not None else [],
            "tools": tools if tools is not None else [],
        }
    ]
    for idx, (name, input_) in enumerate(actions):
        events.append(
            {
                "type": "assistant",
                "message": {
                    "id": f"msg_test_{idx}",
                    "content": [{"type": "tool_use", "name": name, "input": input_}],
                    "usage": usage,
                },
            }
        )
    events.append(
        {
            "type": "result",
            "subtype": "success",
            "num_turns": num_turns,
            "duration_ms": 1234,
            "total_cost_usd": 0.01,
            "usage": usage,
        }
    )
    return events


def _init_and_result_events(
    mcp_servers: list[dict[str, str]], tools: list[str] | None = None
) -> list[dict]:
    return [
        {"type": "system", "subtype": "init", "mcp_servers": mcp_servers, "tools": tools or []},
        {"type": "result", "subtype": "success", "num_turns": 1, "usage": {}},
    ]


def _setup(tmp_path: Path, slug: str, *, variant: str, leg_index: int = 0):
    """Materialize a FRESH workspace under `tmp_path/variant` -- naive and
    fact-using fixtures must never share a workspace directory, or one fixture's
    mutation would leak into the other's `FileUnchanged` baseline.
    """
    scenario = scen.get(slug)
    root = tmp_path / variant
    workspace = root / "workspace"
    verify_dir = root / "verify"
    baselines = scen.materialize(scenario, workspace)
    leg = scenario.legs[leg_index]
    scen.materialize_verifiers(leg, verify_dir)
    return scenario, leg, workspace, verify_dir, baselines


def _write(workspace: Path, rel: str, content: str) -> None:
    target = workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _score(leg, workspace: Path, baselines, verify_dir: Path, events: list[dict]):
    run_score = scorer.score_run(
        leg, workspace=workspace, leg_start_baselines=baselines,
        transcript=events, verify_dir=verify_dir,
    )
    actions = scorer.build_action_stream(events)
    metrics = scorer.leg_metrics(
        leg, events=events, actions=actions, workspace=workspace,
        leg_start_baselines=baselines, k=1, origin="discovered",
    )
    return run_score, metrics


def _assert_discriminates(naive_score, naive_metrics, fact_score, fact_metrics) -> None:
    assert naive_score["fact_used"] is False, naive_score
    assert fact_score["fact_used"] is True, fact_score
    assert naive_metrics["censored"] is True
    assert naive_metrics["turns_to_fact"] is None
    assert fact_metrics["censored"] is False
    assert fact_metrics["turns_to_fact"] is not None


# ---------------------------------------------------------------------------
# cross-scenario structural invariants
# ---------------------------------------------------------------------------


def test_every_leg_declares_a_real_primary_check():
    for slug, scenario in scen.SCENARIOS.items():
        for i, leg in enumerate(scenario.legs, start=1):
            names = [getattr(c, "name") for c in leg.checks]
            assert leg.primary_check in names, (
                f"{slug} leg {i}: primary_check {leg.primary_check!r} not among {names!r}"
            )


def test_note_variant_content_byte_identical_to_fact_sentence():
    """`__post_init__` already enforces this at construction time; this test makes
    the invariant an explicit, independently-checkable assertion (DESIGN.md 11.6)."""
    for slug, scenario in scen.SCENARIOS.items():
        for variant in scenario.note_variants:
            assert scenario.fact_sentence in variant.content, (
                f"{slug}/{variant.variant}: note body drops the byte-identical fact sentence"
            )


def test_forcing_leg_flag_set_only_on_leg_one_of_discovered_scenarios():
    """S1-S4 (discovered) force the mistake via leg 1's task; S5-S6 (told) state the
    fact directly in leg 1's prompt instead and never set `is_forcing_leg` (its own
    docstring: "leg 1 of a DISCOVERED scenario"). Either way, no leg after 1 forces.
    """
    for slug, scenario in scen.SCENARIOS.items():
        if scenario.origin == "discovered":
            assert scenario.legs[0].is_forcing_leg, f"{slug}: leg 1 must be the forcing leg"
        else:
            assert not scenario.legs[0].is_forcing_leg, (
                f"{slug}: a TOLD scenario states the fact directly, no forcing leg"
            )
        for i, leg in enumerate(scenario.legs[1:], start=2):
            assert not leg.is_forcing_leg, f"{slug} leg {i}: only leg 1 may force"


def test_fact_token_confined_to_anchor_files_and_leg1_prompt():
    """No `fact_token` may appear in any scenario file except its designated anchor
    file(s), and in no leg->=2 prompt (DESIGN.md 5.4, 11.6). Leg 1's prompt is exempt:
    S5/S6 (told) state the fact directly there for every arm (shared leg 1).
    """
    for slug, scenario in scen.SCENARIOS.items():
        anchors = set(scenario.anchor_files())
        for rel, content in scenario.files.items():
            if rel in anchors:
                continue
            for token in scenario.fact_tokens:
                assert token not in content, (
                    f"{slug}: fact_token {token!r} leaked into non-anchor file {rel!r}"
                )
        for i, leg in enumerate(scenario.legs[1:], start=2):
            for token in scenario.fact_tokens:
                assert token not in leg.prompt, (
                    f"{slug} leg {i}: fact_token {token!r} leaked into its own prompt"
                )


# ---------------------------------------------------------------------------
# UPG-EVAL-ANCHOR-CONFOUND: t3_metrics's anchor_checked is a cross-arm
# comparison cell -- computed for every variant (including variant=None, arm
# "none"), sourced from the scenario's own anchor_files(), never from
# variant.anchors directly (populated only on the verifiable rung).
# ---------------------------------------------------------------------------


def test_t3_metrics_anchor_checked_computed_for_plain_variant_via_read():
    s1 = scen.get("release_via_ci")
    plain = next(v for v in s1.note_variants if v.variant == "plain")
    events = _transcript([("Read", {"file_path": ".github/workflows/release.yml"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        plain, fact_sentence=s1.fact_sentence, actions=actions,
        scenario_anchors=s1.anchor_files(),
    )
    assert result["anchor_checked"] is True
    # Properties of the PLANTED note itself stay scoped to the verifiable rung --
    # a plain-rung note has no verify hint, so these remain None regardless.
    assert result["verify_command_ran"] is None
    assert result["trail_chars"] is None


def test_t3_metrics_anchor_checked_false_for_plain_variant_when_anchor_untouched():
    s1 = scen.get("release_via_ci")
    plain = next(v for v in s1.note_variants if v.variant == "plain")
    events = _transcript([("Read", {"file_path": "CHANGELOG.md"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        plain, fact_sentence=s1.fact_sentence, actions=actions,
        scenario_anchors=s1.anchor_files(),
    )
    assert result["anchor_checked"] is False


def test_t3_metrics_anchor_checked_computed_for_provenance_variant_via_bash():
    s1 = scen.get("release_via_ci")
    provenance = next(v for v in s1.note_variants if v.variant == "provenance")
    events = _transcript([("Bash", {"command": "cat .github/workflows/release.yml"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        provenance, fact_sentence=s1.fact_sentence, actions=actions,
        scenario_anchors=s1.anchor_files(),
    )
    assert result["anchor_checked"] is True


def test_t3_metrics_anchor_checked_computed_when_no_note_was_planted_at_all():
    """arm "none": variant is None (no note planted), but the scenario's anchor
    is a workspace-level fact-verification artifact independent of whether any
    note named it -- this is the no-memory CONTROL cell the confound removed.
    """
    s1 = scen.get("release_via_ci")
    events = _transcript([("Grep", {"path": ".github/workflows/release.yml", "pattern": "id-token"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence=s1.fact_sentence, actions=actions,
        scenario_anchors=s1.anchor_files(),
    )
    assert result["anchor_checked"] is True
    assert result["verify_command_ran"] is None
    assert result["trail_chars"] is None


def test_t3_metrics_anchor_checked_is_none_for_uncorroborable_scenario():
    """An uncorroborable scenario (S5/S6, told not discovered) declares no
    verifiable rung, so anchor_files() is empty and there is nothing in the
    workspace to check -- anchor_checked stays None (not a false False)."""
    s5 = scen.get("deploy_reverted_by_reconciler")
    assert s5.anchor_files() == ()
    plain = next(v for v in s5.note_variants if v.variant == "plain")
    events = _transcript([("Read", {"file_path": "deploy/queue.yaml"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        plain, fact_sentence=s5.fact_sentence, actions=actions,
        scenario_anchors=s5.anchor_files(),
    )
    assert result["anchor_checked"] is None


# ---------------------------------------------------------------------------
# UPG-EVAL-TOUCHES-ANCHOR-SUBSTRING: anchor matching must be by path component
# (Read/Grep) and by shell token (Bash), never a raw substring search -- a raw
# `a in v` matches a `.bak` sibling, a Grep `pattern` argument that happens to
# contain the anchor's characters, and an `echo` of the anchor's name, none of
# which are the agent actually inspecting the anchor file. These tests use a
# synthetic anchor ("config.py") via `scenario_anchors=` directly rather than a
# real scenario's `anchor_files()` -- none of the six scenarios' real anchors
# happen to exercise the `.bak`/`echo` shapes, and `t3_metrics` takes
# `scenario_anchors` as a plain argument for exactly this reason.
# ---------------------------------------------------------------------------


def test_t3_metrics_anchor_checked_false_for_bak_sibling_path():
    """`vendor/other/config.py.bak` must NOT match anchor `config.py` -- the
    trailing path COMPONENT is `config.py.bak`, not `config.py`; a raw
    substring search matches this (the defect), a component comparison does
    not (the fix)."""
    events = _transcript([("Read", {"file_path": "vendor/other/config.py.bak"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence="irrelevant", actions=actions, scenario_anchors=("config.py",),
    )
    assert result["anchor_checked"] is False


def test_t3_metrics_anchor_checked_false_for_echo_only_bash_command():
    """`echo config.py` must NOT count as touching the anchor -- echo's
    arguments are literal output text, never a file the shell or the invoked
    program opens, unlike the genuine inspection commands
    (`grep`/`head`/`cat`/`python <path>`) every scenario's own verify hint
    uses (see `test_t3_metrics_anchor_checked_computed_for_provenance_variant_via_bash`
    and `test_t3_metrics_verify_command_ran_and_trail_chars_still_verifiable_only`
    below, which must keep scoring True)."""
    events = _transcript([("Bash", {"command": "echo config.py"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence="irrelevant", actions=actions, scenario_anchors=("config.py",),
    )
    assert result["anchor_checked"] is False


def test_t3_metrics_anchor_checked_false_for_grep_pattern_argument_mention():
    """A Grep `pattern` argument that happens to contain the anchor's name is
    not a path reference at all -- `_path_values` already excludes `pattern`
    from consideration (only `file_path`/`path`/`notebook_path` are path
    inputs), so this stays False both before and after the fix; pinned here
    so a future change to `_path_values` cannot silently reintroduce it."""
    events = _transcript(
        [("Grep", {"path": "src", "pattern": "references config.py somewhere"})]
    )
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence="irrelevant", actions=actions, scenario_anchors=("config.py",),
    )
    assert result["anchor_checked"] is False


def test_t3_metrics_anchor_checked_true_for_genuine_subpath_read():
    """A real, deeper path whose trailing component IS the anchor still
    counts -- `src/pkg/config.py` names the same file `config.py` does, just
    from a different root; the fix must not narrow this to exact-string
    equality."""
    events = _transcript([("Read", {"file_path": "src/pkg/config.py"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence="irrelevant", actions=actions, scenario_anchors=("config.py",),
    )
    assert result["anchor_checked"] is True


def test_t3_metrics_anchor_checked_true_for_genuine_bash_inspection_token():
    """A real inspection command (`head -5 config.py`) with the anchor as a
    trailing argument token must still count -- this is the exact shape of
    every scenario's own `verify_hint` (DESIGN.md 7.2)."""
    events = _transcript([("Bash", {"command": "head -5 config.py"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence="irrelevant", actions=actions, scenario_anchors=("config.py",),
    )
    assert result["anchor_checked"] is True


def test_t3_metrics_verify_command_ran_and_trail_chars_still_verifiable_only():
    """Regression guard: the two variant-specific T3 measures are unaffected by
    this fix -- still None for every non-verifiable variant/arm, still computed
    normally for the verifiable rung."""
    s1 = scen.get("release_via_ci")
    verifiable = next(v for v in s1.note_variants if v.variant == "verifiable")
    plain = next(v for v in s1.note_variants if v.variant == "plain")

    events = _transcript([("Bash", {"command": 'grep -n "id-token" .github/workflows/release.yml'})])
    actions = scorer.build_action_stream(events)

    verifiable_result = scorer.t3_metrics(
        verifiable, fact_sentence=s1.fact_sentence, actions=actions,
        scenario_anchors=s1.anchor_files(),
    )
    assert verifiable_result["verify_command_ran"] is True
    assert verifiable_result["trail_chars"] is not None and verifiable_result["trail_chars"] > 0

    plain_result = scorer.t3_metrics(
        plain, fact_sentence=s1.fact_sentence, actions=actions,
        scenario_anchors=s1.anchor_files(),
    )
    assert plain_result["verify_command_ran"] is None
    assert plain_result["trail_chars"] is None


def test_t3_metrics_s1_anchor_checked_is_non_discriminating_across_arms():
    """Documents the KNOWN, NOT-fixed-by-this-change scenario-design confound
    on S1: its own leg-1 forcing step already requires touching
    `.github/workflows/release.yml`, which is also S1's sole anchor, so
    anchor_checked is expected to read True for every arm (including the
    no-memory control) as soon as the forcing step's own Read fires -- this
    scenario's anchor_checked numbers are non-discriminating regardless of
    which variant/arm produced them. A future scenario must keep its anchor
    separable from its forcing step's own required artifact, or this collapse
    repeats (see t3_metrics's own docstring)."""
    s1 = scen.get("release_via_ci")
    leg1 = s1.legs[0]
    assert leg1.is_forcing_leg is True
    assert s1.anchor_files() == (".github/workflows/release.yml",)
    # This is the artifact leg 1's own forcing step (releasing 1.4.0) requires
    # touching, regardless of what note (if any) this leg's arm was handed:
    events = _transcript([("Read", {"file_path": ".github/workflows/release.yml"})])
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence=s1.fact_sentence, actions=actions,
        scenario_anchors=s1.anchor_files(),
    )
    assert result["anchor_checked"] is True  # true even with variant=None / arm "none"


# ---------------------------------------------------------------------------
# UPG-EVAL-S1-ANCHOR-SEPARABILITY: audit of the SAME invariant (t3_metrics's own
# docstring) across every scenario, not only S1. A scenario's anchor must be
# separable from whatever its own forcing step OR fact_acquisition already
# requires touching, or anchor_checked collapses to non-discriminating. This is
# a SCENARIO-DESIGN property, never patched in scorer.py.
#
# Audit result:
#   S1 release_via_ci     -- VIOLATION, leg 1 only (pinned above). Cannot be
#                             fixed without a rebuild: `.github/workflows/
#                             release.yml` is the sole artifact in the repo
#                             documenting the correct release process, so any
#                             from-scratch investigation that corrects the
#                             twine mistake necessarily reads it. Legs 2-4 stay
#                             separable (a memory arm that trusts its note tags
#                             directly, touching nothing).
#   S2 spec_lives_outside -- COMPLIANT at every leg (pinned below). The forcing
#                             step's own `make check` failure output is what a
#                             from-scratch investigation surfaces
#                             (rediscovery_work's `BashAction(r"make\s+check|
#                             docs.?lint")` alternative) -- never a literal read
#                             of docs_lint.py's own source.
#   S3 runner_not_pytest  -- WAS a violation at EVERY leg; REPAIRED by
#                             UPG-EVAL-S3-S4-SCENARIO-REDESIGN and pinned below.
#                             The old anchor (`tools/t`) was the executable
#                             `fact_acquisition` requires running, so acquiring
#                             the fact and touching the anchor were the same
#                             Bash action by construction (`_touches_anchor`
#                             matches a Bash command by shell TOKEN -- running a
#                             script and inspecting it are indistinguishable).
#                             The anchor moved to `.github/workflows/ci.yml`,
#                             which no leg's prompt, primary check, or
#                             fact_acquisition requires touching, and the fact's
#                             in-repo documentation moved with it (the
#                             fact_token confinement test binds documentation
#                             location to anchor location, so there is no
#                             fixture-free repair). NOT repaired scorer-side:
#                             t3_metrics's computation is byte-identical.
#   S4 secrets_not_dotenv -- Same collapse, same repair (pinned below). The old
#                             anchor (`scripts/envctl`) was the executable
#                             `fact_acquisition` requires running, AND it was
#                             not even the file the verifiable rung's own
#                             verify_hint named. The anchor moved to
#                             `src/harbor/config.py` -- the hint's own target,
#                             and the artifact the fact is a claim ABOUT.
#   S5/S6 (told, uncorroborable) -- vacuously compliant: anchor_files() == ()
#                             for both (no verifiable rung, nothing in the
#                             workspace states the fact), so anchor_checked is
#                             `None`, never a non-discriminating `True`
#                             (test_t3_metrics_anchor_checked_is_none_for_
#                             uncorroborable_scenario covers S5; extended below
#                             to confirm S6 identically).
# ---------------------------------------------------------------------------


def test_t3_metrics_s2_anchor_checked_false_for_a_no_recall_control():
    """S2's no-recall control (arm A's own expected leg-1 shape, DESIGN.md:
    'Edits orbit/docs/spec.md directly; make check still fails at the end.')
    runs `make check`, sees it fail, and edits the wrong mirror file --
    without ever reading docs_lint.py's own source -- and still scores
    anchor_checked False. This is the documented no-recall control the audit
    above requires: S2 stays separable, unlike S1/S3/S4."""
    s2 = scen.get("spec_lives_outside")
    assert s2.anchor_files() == ("orbit/tools/docs_lint.py",)
    events = _transcript(
        [
            ("Bash", {"command": "make check"}),
            ("Edit", {"file_path": "orbit/docs/spec.md", "old_string": "x", "new_string": "y"}),
        ]
    )
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence=s2.fact_sentence, actions=actions,
        scenario_anchors=s2.anchor_files(),
    )
    assert result["anchor_checked"] is False


def test_t3_metrics_s3_anchor_separable_from_fact_acquisition_every_leg():
    r"""UPG-EVAL-S3-S4-SCENARIO-REDESIGN, S3: the repaired invariant, pinned as
    the exact inverse of the collapse it replaces. Before the repair the anchor
    was `tools/t`, the very executable `fact_acquisition` requires running
    (`BashAction(r"(\./)?tools/t\b")`), so the canonical fact-acquisition action
    scored anchor_checked True at every leg and the cell was a relabelling of
    `fact_acquired`. With the anchor at `.github/workflows/ci.yml`, the same
    canonical action acquires the fact and touches NOTHING -- so a True here can
    only come from a separate, chosen corroboration act, which is what DESIGN.md
    7.2's Q1 reads it as."""
    s3 = scen.get("runner_not_pytest")
    assert s3.anchor_files() == (".github/workflows/ci.yml",)
    for i, leg in enumerate(s3.legs, start=1):
        events = _transcript([("Bash", {"command": "./tools/t"})])
        actions = scorer.build_action_stream(events)
        assert scorer.first_index(actions, leg.fact_acquisition) is not None, f"leg {i}"
        result = scorer.t3_metrics(
            None, fact_sentence=s3.fact_sentence, actions=actions,
            scenario_anchors=s3.anchor_files(),
        )
        assert result["anchor_checked"] is False, f"leg {i}: fact acquired must not imply anchor"

    # ...and the corroboration act, which no leg requires, still registers.
    verifiable = next(v for v in s3.note_variants if v.variant == "verifiable")
    checked = scorer.build_action_stream(
        _transcript([("Bash", {"command": verifiable.verify_hint})])
    )
    result = scorer.t3_metrics(
        verifiable, fact_sentence=s3.fact_sentence, actions=checked,
        scenario_anchors=s3.anchor_files(),
    )
    assert result["anchor_checked"] is True
    assert result["verify_command_ran"] is True


def test_t3_metrics_s3_anchor_checked_false_for_a_no_recall_control():
    """S3's documented no-recall control, the acceptance bar this scenario owes
    (matching S2's own, `test_t3_metrics_s2_anchor_checked_false_for_a_no_recall
    _control`): arm "none" walking its declared `arm_a_expectation` -- "Runs bare
    pytest, sees nine collection errors, and starts silencing legacy/ instead of
    using tools/t" -- commits the mistake and scores anchor_checked False. Under
    the pre-repair anchor this control was unreachable for any leg in which the
    agent ever reached `tools/t`."""
    s3 = scen.get("runner_not_pytest")
    leg1 = s3.legs[0]
    events = _transcript(
        [
            ("Bash", {"command": "python -m pytest -q"}),
            ("Read", {"file_path": "legacy/test_native_a.py"}),
            (
                "Edit",
                {
                    "file_path": "pytest.ini",
                    "old_string": "addopts = -q",
                    "new_string": "addopts = -q\nnorecursedirs = legacy",
                },
            ),
        ]
    )
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence=s3.fact_sentence, actions=actions,
        scenario_anchors=s3.anchor_files(),
    )
    assert result["anchor_checked"] is False
    # ...and it is a real no-recall control: the mistake fired, the fact never landed.
    assert scorer.first_index(actions, leg1.mistake_signature) is not None
    assert scorer.first_index(actions, leg1.fact_acquisition) is None


def test_t3_metrics_s4_anchor_separable_from_fact_acquisition_every_leg():
    """Same repair as S3 (see that test's docstring): S4's anchor moved off the
    executable `fact_acquisition` requires running (`scripts/envctl`) and onto
    `src/harbor/config.py`, which is also the file the verifiable rung's own
    verify_hint always named -- so `anchor_checked` and `verify_command_ran` now
    measure the same artifact instead of two different ones."""
    s4 = scen.get("secrets_not_dotenv")
    assert s4.anchor_files() == ("src/harbor/config.py",)
    for i, leg in enumerate(s4.legs, start=1):
        events = _transcript([("Bash", {"command": "scripts/envctl get SIGNING_KEY"})])
        actions = scorer.build_action_stream(events)
        assert scorer.first_index(actions, leg.fact_acquisition) is not None, f"leg {i}"
        result = scorer.t3_metrics(
            None, fact_sentence=s4.fact_sentence, actions=actions,
            scenario_anchors=s4.anchor_files(),
        )
        assert result["anchor_checked"] is False, f"leg {i}: fact acquired must not imply anchor"

    verifiable = next(v for v in s4.note_variants if v.variant == "verifiable")
    checked = scorer.build_action_stream(
        _transcript([("Bash", {"command": verifiable.verify_hint})])
    )
    result = scorer.t3_metrics(
        verifiable, fact_sentence=s4.fact_sentence, actions=checked,
        scenario_anchors=s4.anchor_files(),
    )
    assert result["anchor_checked"] is True
    assert result["verify_command_ran"] is True


def test_t3_metrics_s4_anchor_checked_false_for_a_no_recall_control():
    """S4's documented no-recall control, the acceptance bar this scenario owes:
    arm "none" walking the `.env` half of its declared `arm_a_expectation`
    ("Adds the new key to .env and/or a dotenv loader to config.py") -- reads the
    decoy, writes the decoy, runs `make run`, and never opens config.py. Scores
    anchor_checked False with the mistake fired and the fact never acquired.

    The OTHER half of that expectation (adding a loader to config.py) does read
    the anchor first, so a loader-branch control would score anchor_checked True.
    That is not a confound: it is an agent that inspected the ground truth and
    contradicted it anyway, and the two branches are distinguishable because
    `mistake_committed` fires in both while `anchor_checked` separates them."""
    s4 = scen.get("secrets_not_dotenv")
    leg1 = s4.legs[0]
    events = _transcript(
        [
            ("Read", {"file_path": ".env"}),
            (
                "Edit",
                {
                    "file_path": ".env",
                    "old_string": "ACME_API_KEY=REPLACE_ME_LOCAL_ONLY",
                    "new_string": "ACME_API_KEY=REPLACE_ME_LOCAL_ONLY\nACME_WEBHOOK_SECRET=abc123",
                },
            ),
            ("Bash", {"command": "make run"}),
        ]
    )
    actions = scorer.build_action_stream(events)
    result = scorer.t3_metrics(
        None, fact_sentence=s4.fact_sentence, actions=actions,
        scenario_anchors=s4.anchor_files(),
    )
    assert result["anchor_checked"] is False
    assert scorer.first_index(actions, leg1.mistake_signature) is not None
    assert scorer.first_index(actions, leg1.fact_acquisition) is None


def test_t3_metrics_s5_and_s6_anchor_checked_vacuously_compliant():
    """S5/S6 (told, uncorroborable) declare no verifiable rung -- nothing in
    the workspace states the fact for either -- so anchor_files() is empty and
    the separability invariant is vacuously satisfied: anchor_checked is
    `None` (nothing to check), never a non-discriminating `True`."""
    s5 = scen.get("deploy_reverted_by_reconciler")
    s6 = scen.get("bench_box_only")
    assert s5.anchor_files() == ()
    assert s6.anchor_files() == ()
    for scenario in (s5, s6):
        events = _transcript([("Bash", {"command": "cat some/file"})])
        actions = scorer.build_action_stream(events)
        result = scorer.t3_metrics(
            None, fact_sentence=scenario.fact_sentence, actions=actions,
            scenario_anchors=scenario.anchor_files(),
        )
        assert result["anchor_checked"] is None


def test_verify_scripts_never_land_inside_the_workspace(tmp_path):
    """A verifier inside the workspace would leak the answer to the no-memory arm."""
    for slug, scenario in scen.SCENARIOS.items():
        workspace = tmp_path / slug / "workspace"
        verify_dir = tmp_path / slug / "verify"
        scen.materialize(scenario, workspace)
        for leg in scenario.legs:
            scen.materialize_verifiers(leg, verify_dir)
            for name in leg.verify_scripts:
                assert not (workspace / name).exists()
                assert not any(p.name == Path(name).name for p in workspace.rglob("*"))


# ---------------------------------------------------------------------------
# DEFECT 10 (direction 1, user decision 2026-07-30; DESIGN.md 6.5): per-leg reset
# of scenario-declared `critical_residue_paths`.
# ---------------------------------------------------------------------------


def test_critical_residue_paths_entry_must_have_a_matching_files_seed():
    """`LongitudinalScenario.__post_init__` refuses a declared reset path with no
    `files` seed to restore FROM -- `run_leg.py`'s `_apply_critical_residue_reset`
    trusts this invariant instead of re-checking it at run time (KeyError would be
    the alternative, much later and far from the authoring mistake).
    """
    real = scen.SCENARIOS["deploy_reverted_by_reconciler"]
    with pytest.raises(ValueError, match="critical_residue_paths"):
        dataclasses.replace(real, critical_residue_paths=("deploy/queue.yaml", "no/such/file.txt"))


def test_no_scenario_declares_critical_residue_paths_by_accident():
    """S1-S4 each target a distinct value/artifact per leg, so no leg's check can be
    pre-satisfied by an earlier leg's own residue. S5 (`deploy/queue.yaml`) and S6
    (`RESULTS.md`, UPG-EVAL-S6-LEG4-VACUOUS: leg 4 re-measures leg 1's algorithm on
    a deterministic bench box) do need a reset. This is a deliberate allow-list, not
    an oversight -- a future scenario that DOES need one must add itself here.
    """
    declared = {slug: s.critical_residue_paths for slug, s in scen.SCENARIOS.items() if s.critical_residue_paths}
    assert declared == {
        "deploy_reverted_by_reconciler": ("deploy/queue.yaml",),
        "bench_box_only": ("RESULTS.md",),
    }


def test_s5_queue_gained_staging_entry_is_uniform_minimum_one_across_every_leg():
    """Post-fix: every leg's engagement half is a flat minimum=1 (not the old
    cumulative 1/2/3/4) -- coherent only because `deploy/queue.yaml` is reset to
    its zero-staging-entry seed at every k>=2 leg's start (`critical_residue_paths`),
    so ANY staging entry present at leg-end was necessarily added THIS leg.
    """
    s5 = scen.SCENARIOS["deploy_reverted_by_reconciler"]
    for i, leg in enumerate(s5.legs, start=1):
        allof = leg.checks[0]
        sub = next(c for c in allof.of if c.name == "queue_gained_staging_entry")
        assert sub.minimum == 1, f"leg {i}: expected uniform minimum=1, got {sub.minimum}"


def test_s5_check_measures_this_legs_own_addition_not_leg1_residue(tmp_path):
    """Mechanical proof of the check-semantics fix, via the real `evaluate_check`:
    starting from the scenario seed (what every k>=2 leg sees after
    `_apply_critical_residue_reset`), the engagement half fails until THIS leg adds
    its own entry, and passes as soon as it does -- for every leg, not just leg 1.
    Was previously only true for leg 1 (legs 2-4 needed 2/3/4 CUMULATIVE entries,
    so leg 1's leftover alone already satisfied leg 2's old check).
    """
    s5 = scen.SCENARIOS["deploy_reverted_by_reconciler"]
    for i, leg in enumerate(s5.legs, start=1):
        workspace = tmp_path / f"leg{i}"
        baselines = scen.materialize(s5, workspace)
        verify_dir = tmp_path / f"verify{i}"

        # Seed only (post-reset leg-start state, or leg 1's own untouched start):
        # the engagement half must fail -- nothing this leg did yet.
        pre = scorer.evaluate_check(
            leg.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
        )
        engagement_pre = next(s for s in pre["sub_checks"] if s["name"] == "queue_gained_staging_entry")
        assert engagement_pre["passed"] is False, f"leg {i}: seed alone must not satisfy the check"

        # This leg's own compliant addition (append, never edit in place, per the
        # seed file's own header comment) -- must now satisfy it.
        _write(
            workspace, "deploy/queue.yaml",
            s5.files["deploy/queue.yaml"] + '- date: "2026-07-20"\n  target: staging\n  ref: HEAD\n',
        )
        post = scorer.evaluate_check(
            leg.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
        )
        engagement_post = next(s for s in post["sub_checks"] if s["name"] == "queue_gained_staging_entry")
        assert engagement_post["passed"] is True, f"leg {i}: own addition must satisfy the check"


# ---------------------------------------------------------------------------
# DEFECT 12 (Option A, user decision 2026-08-03): S5's hidden fact has a
# descriptive half (the reconciler silently reverts a direct `deploy.sh` run) and
# a prescriptive half (`deploy/queue.yaml` is the sanctioned channel). The
# `deploy/README.md` fixture used to leak the prescriptive half ("queue.yaml is
# consumed by the release bot"), letting a fixture-reading agent derive the
# correct action with no note. It is now mechanical only; the prescriptive half
# lives solely in the note variants and the fact sentence.
# ---------------------------------------------------------------------------


def test_s5_readme_fixture_is_de_prescribed():
    """`deploy/README.md` must name `deploy/queue.yaml`'s fields (what a mechanical
    reader would find) without ever stating who consumes the file or that it is the
    deploy channel -- that claim would let an agent derive the correct action
    (queue.yaml, not deploy.sh) from the workspace alone, with no note, collapsing
    the action contrast the note-variant ladder (DESIGN.md section 7.2) measures.
    """
    s5 = scen.SCENARIOS["deploy_reverted_by_reconciler"]
    readme = s5.files["deploy/README.md"]
    assert readme == "deploy/queue.yaml holds a list of dated entries: date, target, ref, requested_by.\n"
    queue_seed = s5.files["deploy/queue.yaml"]
    for leaked_phrase in ("consumed by", "release bot", "release-bot", "sanctioned", "bot picks it up"):
        assert leaked_phrase not in readme, f"README fixture still leaks {leaked_phrase!r}"
        assert leaked_phrase.lower() not in queue_seed.lower(), (
            f"queue seed still leaks {leaked_phrase!r} -- the prescriptive half must "
            "live only in the note variants and the fact sentence"
        )


def test_s5_queue_seed_bytes_are_pinned():
    """`deploy/queue.yaml`'s seed is declared in `critical_residue_paths` (DEFECT 10,
    DESIGN.md section 6.5) and `run_leg.py` restores it to exactly this string at the
    start of every k>=2 leg. A byte change here -- even whitespace -- would silently
    change the workspace surface mid-trajectory for any recorded S5 trajectory a
    later leg extends (T5's leg-4 extension, T2's reuse of a shared leg 1), so this
    string is pinned byte-for-byte rather than just checked for the fields it
    contains. The seed's former "Consumed by the release bot" first line carried the
    same prescriptive leak as the README; it was de-prescribed at the SECOND DEFECT
    12 boundary, which landed only after the last pre-fix S5 extension had run, and
    the seed is pinned again from that commit on.
    """
    s5 = scen.SCENARIOS["deploy_reverted_by_reconciler"]
    assert s5.files["deploy/queue.yaml"] == (
        "# Entries are appended, never edited in place.\n"
        "- date: \"2026-07-01\"\n"
        "  target: prod-canary\n"
        "  ref: v1.1.0\n"
        "  requested_by: ops-bot\n"
    )


# ---------------------------------------------------------------------------
# UPG-EVAL-S6-LEG4-VACUOUS: S6's leg 4 re-measures leg 1's algorithm on a
# deterministic bench box, so its primary check was satisfied by leg 1's residue
# with zero leg-4 work. Fixed by declaring `RESULTS.md` in the same
# `critical_residue_paths` mechanism S5 uses (DESIGN.md 6.5).
# ---------------------------------------------------------------------------

_S6_RADIX_JSON = '{"algorithm": "radix_sort", "n": 1000000, "ns_op": 182, "source": "remote-box"}\n'
_S6_RADIX_ROW = "| radix_sort | 1000000 | 182 | remote-box |\n"

# The scenario's own deterministic box numbers (`bench/boxrun.sh`), keyed by the
# algorithm each leg's primary check names.
_S6_BOX_NS = {"radix_sort": 182, "merge_sort": 241, "tim_sort": 205}
_S6_LEG_ALGO = ("radix_sort", "merge_sort", "tim_sort", "radix_sort")


def _s6_box_run(workspace: Path, algo: str) -> None:
    """What a compliant `make bench REMOTE=1 ALGO=<algo>` leaves behind: the box's
    result JSON, plus the row the agent pastes into RESULTS.md from it."""
    _write(
        workspace, f"bench/results/remote-{algo}.json",
        f'{{"algorithm": "{algo}", "n": 1000000, "ns_op": {_S6_BOX_NS[algo]}, "source": "remote-box"}}\n',
    )
    results = (workspace / "RESULTS.md").read_text(encoding="utf-8")
    _write(workspace, "RESULTS.md", results + f"| {algo} | 1000000 | {_S6_BOX_NS[algo]} | remote-box |\n")


def test_s6_leg4_check_is_vacuous_against_leg1_residue_and_non_vacuous_after_the_reset(tmp_path):
    """The acceptance case, both directions, through the real `evaluate_check`.

    Leg 4's `VerifyCommand` argv is leg 1's verbatim (`verify_traceability.py
    radix_sort`) and `boxrun.sh` is deterministic, so leg 1's compliant end state
    already exits it 0: an agent that does NOTHING at leg 4 scores a pass. Declaring
    `RESULTS.md` in `critical_residue_paths` -- which `run_leg.py` applies at the
    start of every k>=2 leg -- is what makes the leg 4 verdict its own again.
    """
    s6 = scen.SCENARIOS["bench_box_only"]
    leg4 = s6.legs[3]
    workspace = tmp_path / "ws"
    baselines = scen.materialize(s6, workspace)
    verify_dir = tmp_path / "verify"
    scen.materialize_verifiers(leg4, verify_dir)

    # Leg 1's compliant end state (the residue leg 4 inherits without a reset).
    _write(workspace, "bench/results/remote-radix_sort.json", _S6_RADIX_JSON)
    _write(workspace, "RESULTS.md", s6.files["RESULTS.md"] + _S6_RADIX_ROW)

    unreset = scorer.evaluate_check(
        leg4.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
    )
    assert unreset["passed"] is True, (
        "fixture no longer reproduces the defect: leg 1's residue must be what made "
        "leg 4's check vacuous"
    )

    # The declared reset, applied exactly as `_apply_critical_residue_reset` does:
    # restore the declared path to its `files` seed, nothing else.
    assert s6.critical_residue_paths == ("RESULTS.md",)
    for path in s6.critical_residue_paths:
        _write(workspace, path, s6.files[path])

    reset = scorer.evaluate_check(
        leg4.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
    )
    assert reset["passed"] is False, "post-reset, a leg 4 that does nothing must not pass"

    # ...and leg 4's own compliant box run passes it again.
    _s6_box_run(workspace, "radix_sort")
    own_work = scorer.evaluate_check(
        leg4.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
    )
    assert own_work["passed"] is True, "leg 4's own bench-box row must satisfy its check"


def test_s6_every_leg_check_measures_its_own_row_from_the_reset_seed(tmp_path):
    """Blast-radius guard for the reset: legs 2 and 3 now also start from the seed
    table, so this pins that each leg's primary check still fails on the seed alone
    and passes on that leg's OWN compliant box row -- the reset never makes a
    compliant leg unpassable (the two pre-existing seed rows stay traceable via the
    `remote-quick_sort.json`/`remote-bubble_sort.json` seeds).
    """
    s6 = scen.SCENARIOS["bench_box_only"]
    for i, (leg, algo) in enumerate(zip(s6.legs, _S6_LEG_ALGO), start=1):
        workspace = tmp_path / f"leg{i}"
        baselines = scen.materialize(s6, workspace)
        verify_dir = tmp_path / f"verify{i}"
        scen.materialize_verifiers(leg, verify_dir)

        pre = scorer.evaluate_check(
            leg.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
        )
        assert pre["passed"] is False, f"leg {i}: the reset seed alone must not satisfy the check"

        _s6_box_run(workspace, algo)
        post = scorer.evaluate_check(
            leg.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
        )
        assert post["passed"] is True, f"leg {i}: this leg's own bench-box row must satisfy the check"


def test_s6_laptop_number_still_fails_the_traceability_check(tmp_path):
    """Discrimination, unchanged by the reset: a plain local `make bench` number
    pasted into RESULTS.md has no `bench/results/remote-*.json` to trace to."""
    s6 = scen.SCENARIOS["bench_box_only"]
    leg4 = s6.legs[3]
    workspace = tmp_path / "ws"
    baselines = scen.materialize(s6, workspace)
    verify_dir = tmp_path / "verify"
    scen.materialize_verifiers(leg4, verify_dir)

    _write(workspace, "RESULTS.md", s6.files["RESULTS.md"] + "| radix_sort | 1000000 | 1470 | remote-box |\n")
    verdict = scorer.evaluate_check(
        leg4.checks[0], workspace=workspace, baselines=baselines, commands=(), verify_dir=verify_dir,
    )
    assert verdict["passed"] is False
    assert "untraceable" in verdict["detail"]


def _declared_check(checks, name: str):
    """The declared check object called `name`, searched recursively through any
    `AllOf.of` -- the object-level mirror of `scorer._find_check_by_name`, which
    does the same search over already-evaluated result dicts."""
    for check in checks:
        if getattr(check, "name", None) == name:
            return check
        found = _declared_check(getattr(check, "of", ()), name)
        if found is not None:
            return found
    return None


def _check_shape(value):
    """Structural identity of a declared check, ignoring its `name` (which carries
    a per-leg suffix like `_leg1` and so hides otherwise identical checks)."""
    if dataclasses.is_dataclass(value):
        return (type(value).__name__,) + tuple(
            _check_shape(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.name != "name"
        )
    if isinstance(value, (list, tuple)):
        return tuple(_check_shape(v) for v in value)
    return value


def test_a_leg_check_repeated_verbatim_by_a_later_leg_requires_a_residue_reset():
    """Drift guard, mechanical and scenario-agnostic: two legs whose primary checks
    are structurally identical (same primitive, same parameters -- only the leg
    suffix in `name` differs) ask the workspace the SAME question, so the later
    leg's verdict is pre-satisfiable by the earlier leg's residue unless the
    scenario resets the artifact that answers it. Catches a future leg re-targeting
    an earlier leg's value (how UPG-EVAL-S6-LEG4-VACUOUS arose) without waiting for
    a paid run to produce an always-passing leg.
    """
    for slug, scenario in scen.SCENARIOS.items():
        shapes: dict[object, int] = {}
        for i, leg in enumerate(scenario.legs, start=1):
            primary = _declared_check(leg.checks, leg.primary_check)
            assert primary is not None, f"{slug}: leg {i} names a primary_check it does not declare"
            shape = _check_shape(primary)
            if shape in shapes:
                assert scenario.critical_residue_paths, (
                    f"{slug}: leg {i}'s primary check is identical to leg "
                    f"{shapes[shape]}'s, so leg {i} can pass on leg {shapes[shape]}'s "
                    f"residue alone -- declare the artifact it reads in "
                    f"critical_residue_paths (DESIGN.md 6.5)"
                )
            shapes.setdefault(shape, i)


def _is_path_identifier_char(ch: str) -> bool:
    """Mirrors `agent.working_context_store._store._is_path_identifier_char`
    conservatively: a char that extends a filename/identifier token."""
    return ch.isalnum() or ch in ("_", "-")


def _path_boundary_match(text: str, needle: str) -> bool:
    """Mirrors `agent.working_context_store._store._path_boundary_match`
    conservatively -- True if `needle` occurs in `text` at a genuine path
    boundary, not merely as a substring of a longer identifier (e.g.
    "gate.py" must not false-match inside "uv_regate.py")."""
    if not needle:
        return False
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not _is_path_identifier_char(text[idx - 1])
        after_idx = idx + len(needle)
        after_ok = after_idx >= len(text) or not _is_path_identifier_char(text[after_idx])
        if before_ok and after_ok:
            return True
        start = idx + 1


def _probe_file_structurally_reaches(probe_file: str, variant: "scen.NoteVariant") -> bool:
    """True if `probe_file` would surface `variant` through the daemon's real
    `recall_for_path()` rule: a path-boundary match of the probe file's
    basename or full (workspace-relative) path in the note's content, or an
    exact declared `anchors` entry -- see `NoteVariant`'s own docstring
    invariant. `recall_for_path()` never consults `trigger_paths`
    (UPG-TRIGGERS-INERT-ON-PROXY-STRUCTURAL), so `trigger_paths` is
    deliberately not consulted here either."""
    if probe_file in variant.anchors:
        return True
    basename = probe_file.rsplit("/", 1)[-1]
    return _path_boundary_match(variant.content, basename) or _path_boundary_match(
        variant.content, probe_file
    )


def test_every_leg_probe_file_structurally_reaches_every_note_variant():
    """Regression guard for the T0 probe-reachability class of authoring bug
    (`run_leg.py`'s `_proactive_probe` turn 2 POSTs a leg's `probe_files` to
    `/v1/proactive`, whose structural channel surfaces a note only via a path
    mentioned in the note TEXT or a declared `anchors` entry, never via
    `trigger_paths`): for every scenario, every leg, every note variant, at
    least one of that leg's `probe_files` must structurally reach that
    variant, or `run_plan.py --tier T0` ABORTs at that leg regardless of the
    note's semantic content. Caught release_via_ci legs 2-4 (probe_files
    named no path the fact sentence mentioned) and bench_box_only (probe_files
    named only boxrun.sh, which the fact sentence never mentions) before this
    test existed.
    """
    for slug, scenario in scen.SCENARIOS.items():
        for i, leg in enumerate(scenario.legs, start=1):
            for variant in scenario.note_variants:
                assert any(
                    _probe_file_structurally_reaches(pf, variant) for pf in leg.probe_files
                ), (
                    f"{slug} leg {i}/{variant.variant}: none of {leg.probe_files!r} is "
                    f"mentioned (path-boundary) or anchored in the note content -- "
                    f"structurally unreachable, T0 would ABORT here"
                )


# ---------------------------------------------------------------------------
# arm-blindness of the outcome verdict (DESIGN.md 6.1/6.6)
# ---------------------------------------------------------------------------


def test_score_run_and_leg_metrics_take_no_arm_argument():
    assert "arm" not in inspect.signature(scorer.score_run).parameters
    assert "arm" not in inspect.signature(scorer.leg_metrics).parameters
    # `leg_non_vacuity` is deliberately the ONE place arm identity is consulted.
    assert "arm" in inspect.signature(scorer.leg_non_vacuity).parameters


# ---------------------------------------------------------------------------
# S1 release_via_ci -- discovered, corroborable
# ---------------------------------------------------------------------------


def test_release_via_ci_discriminates(tmp_path):
    naive_scenario, leg, naive_ws, naive_verify, naive_baselines = _setup(
        tmp_path, "release_via_ci", variant="naive"
    )
    naive_events = _transcript(
        [
            ("Bash", {"command": "python -m build"}),
            ("Bash", {"command": "twine upload dist/*"}),
        ]
    )
    naive_score, naive_metrics = _score(leg, naive_ws, naive_baselines, naive_verify, naive_events)

    fact_scenario, _, fact_ws, fact_verify, fact_baselines = _setup(
        tmp_path, "release_via_ci", variant="fact_using"
    )
    _write(fact_ws, "pyproject.toml", fact_scenario.files["pyproject.toml"].replace("1.3.2", "1.4.0"))
    _write(
        fact_ws, "src/ledgerkit/__init__.py",
        fact_scenario.files["src/ledgerkit/__init__.py"].replace("1.3.2", "1.4.0"),
    )
    _write(
        fact_ws, "CHANGELOG.md",
        fact_scenario.files["CHANGELOG.md"].replace("## Unreleased", "## 1.4.0"),
    )
    fact_events = _transcript(
        [
            ("Bash", {"command": "git tag -a v1.4.0 -m 'release 1.4.0'"}),
            ("Bash", {"command": "git push origin v1.4.0 --tags"}),
        ]
    )
    fact_score, fact_metrics = _score(leg, fact_ws, fact_baselines, fact_verify, fact_events)

    _assert_discriminates(naive_score, naive_metrics, fact_score, fact_metrics)
    assert naive_metrics["mistake_committed"] is True
    assert fact_metrics["mistake_committed"] is False


# ---------------------------------------------------------------------------
# S2 spec_lives_outside -- discovered, corroborable
# ---------------------------------------------------------------------------


def test_spec_lives_outside_discriminates(tmp_path):
    naive_scenario, leg, naive_ws, naive_verify, naive_baselines = _setup(
        tmp_path, "spec_lives_outside", variant="naive"
    )
    _write(
        naive_ws, "orbit/docs/spec.md",
        naive_scenario.files["orbit/docs/spec.md"] + "\n## Retry policy\n\nExponential backoff.\n",
    )
    naive_events = _transcript(
        [
            (
                "Edit",
                {
                    "file_path": "orbit/docs/spec.md",
                    "old_string": "Rate limits",
                    "new_string": "Rate limits\n\n## Retry policy\n\nExponential backoff.",
                },
            ),
        ]
    )
    naive_score, naive_metrics = _score(leg, naive_ws, naive_baselines, naive_verify, naive_events)

    fact_scenario, _, fact_ws, fact_verify, fact_baselines = _setup(
        tmp_path, "spec_lives_outside", variant="fact_using"
    )
    _write(
        fact_ws, "orbit-docs/spec.md",
        fact_scenario.files["orbit-docs/spec.md"]
        + "\n## Retry policy\n\nRetries use exponential backoff (see src/orbit/limits.py).\n",
    )
    fact_events = _transcript(
        [
            (
                "Edit",
                {
                    "file_path": "orbit-docs/spec.md",
                    "old_string": "Rate limits",
                    "new_string": "Rate limits\n\n## Retry policy\n\nExponential backoff.",
                },
            ),
        ]
    )
    fact_score, fact_metrics = _score(leg, fact_ws, fact_baselines, fact_verify, fact_events)

    _assert_discriminates(naive_score, naive_metrics, fact_score, fact_metrics)
    assert naive_metrics["mistake_committed"] is True
    assert fact_metrics["mistake_committed"] is False


# ---------------------------------------------------------------------------
# S3 runner_not_pytest -- discovered, corroborable
# ---------------------------------------------------------------------------


def test_runner_not_pytest_discriminates(tmp_path):
    naive_scenario, leg, naive_ws, naive_verify, naive_baselines = _setup(
        tmp_path, "runner_not_pytest", variant="naive"
    )
    _write(
        naive_ws, "pytest.ini",
        naive_scenario.files["pytest.ini"].replace("addopts = -q", "addopts = -q\nnorecursedirs = legacy"),
    )
    naive_events = _transcript(
        [
            (
                "Edit",
                {
                    "file_path": "pytest.ini",
                    "old_string": "addopts = -q",
                    "new_string": "addopts = -q\nnorecursedirs = legacy",
                },
            ),
            ("Bash", {"command": "pytest -q"}),
        ]
    )
    naive_score, naive_metrics = _score(leg, naive_ws, naive_baselines, naive_verify, naive_events)

    _, _, fact_ws, fact_verify, fact_baselines = _setup(
        tmp_path, "runner_not_pytest", variant="fact_using"
    )
    _write(
        fact_ws, "tests/unit/test_rows.py",
        "from feldspar.rows import normalize_row\n\n\n"
        "def test_normalize_row():\n"
        "    assert normalize_row({'a': ' x '}) == {'a': 'x'}\n",
    )
    fact_events = _transcript(
        [
            ("Write", {"file_path": "tests/unit/test_rows.py", "content": "..."}),
            ("Bash", {"command": "./tools/t"}),
        ]
    )
    fact_score, fact_metrics = _score(leg, fact_ws, fact_baselines, fact_verify, fact_events)

    _assert_discriminates(naive_score, naive_metrics, fact_score, fact_metrics)
    assert naive_metrics["mistake_committed"] is True
    assert fact_metrics["mistake_committed"] is False


# ---------------------------------------------------------------------------
# S4 secrets_not_dotenv -- discovered, corroborable
# ---------------------------------------------------------------------------


def test_secrets_not_dotenv_discriminates(tmp_path):
    naive_scenario, leg, naive_ws, naive_verify, naive_baselines = _setup(
        tmp_path, "secrets_not_dotenv", variant="naive"
    )
    _write(naive_ws, ".env", naive_scenario.files[".env"] + "ACME_WEBHOOK_SECRET=abc123\n")
    naive_events = _transcript(
        [
            (
                "Edit",
                {
                    "file_path": ".env",
                    "old_string": "ACME_API_KEY=REPLACE_ME_LOCAL_ONLY",
                    "new_string": "ACME_API_KEY=REPLACE_ME_LOCAL_ONLY\nACME_WEBHOOK_SECRET=abc123",
                },
            ),
        ]
    )
    naive_score, naive_metrics = _score(leg, naive_ws, naive_baselines, naive_verify, naive_events)

    _, _, fact_ws, fact_verify, fact_baselines = _setup(
        tmp_path, "secrets_not_dotenv", variant="fact_using"
    )
    _write(
        fact_ws, "vault/secrets.json",
        '{\n  "ACME_WEBHOOK_SECRET": "wh-value-123",\n  "SIGNING_KEY": "s3cr3t-baseline-value"\n}\n',
    )
    fact_events = _transcript(
        [("Bash", {"command": "scripts/envctl add ACME_WEBHOOK_SECRET wh-value-123"})]
    )
    fact_score, fact_metrics = _score(leg, fact_ws, fact_baselines, fact_verify, fact_events)

    _assert_discriminates(naive_score, naive_metrics, fact_score, fact_metrics)
    assert naive_metrics["mistake_committed"] is True
    assert fact_metrics["mistake_committed"] is False


# ---------------------------------------------------------------------------
# S5 deploy_reverted_by_reconciler -- told, uncorroborable
#
# Leg 1's prompt states the fact directly for every arm (shared leg 1), so leg 2
# is used here: an agent must draw on what was established, not read it fresh.
# ---------------------------------------------------------------------------


def test_deploy_reverted_by_reconciler_discriminates(tmp_path):
    naive_scenario, leg, naive_ws, naive_verify, naive_baselines = _setup(
        tmp_path, "deploy_reverted_by_reconciler", variant="naive", leg_index=1
    )
    _write(naive_ws, ".deploy-state", "2026-07-06T00:00:00Z target=staging\n")
    naive_events = _transcript([("Bash", {"command": "./deploy.sh staging"})])
    naive_score, naive_metrics = _score(leg, naive_ws, naive_baselines, naive_verify, naive_events)

    fact_scenario, _, fact_ws, fact_verify, fact_baselines = _setup(
        tmp_path, "deploy_reverted_by_reconciler", variant="fact_using", leg_index=1
    )
    extra = (
        '- date: "2026-07-05"\n  target: staging\n  ref: v1.1.1\n  requested_by: harness\n'
        '- date: "2026-07-06"\n  target: staging\n  ref: v1.1.2\n  requested_by: harness\n'
    )
    _write(fact_ws, "deploy/queue.yaml", fact_scenario.files["deploy/queue.yaml"] + extra)
    fact_events = _transcript(
        [
            (
                "Edit",
                {
                    "file_path": "deploy/queue.yaml",
                    "old_string": "requested_by: ops-bot",
                    "new_string": "requested_by: ops-bot\n" + extra,
                },
            ),
        ]
    )
    fact_score, fact_metrics = _score(leg, fact_ws, fact_baselines, fact_verify, fact_events)

    _assert_discriminates(naive_score, naive_metrics, fact_score, fact_metrics)
    assert naive_metrics["mistake_committed"] is True
    assert fact_metrics["mistake_committed"] is False


# ---------------------------------------------------------------------------
# S6 bench_box_only -- told, uncorroborable
#
# Leg 1's prompt states the fact directly for every arm (shared leg 1); leg 2 is
# used here for the same reason as S5. This scenario's primary check is a real
# `VerifyCommand` (DESIGN.md's own S6 gap noted in scenarios.py: `mistake_signature`
# is an approximation, but `fact_used`/primary_check is exact and IS exercised here
# via a real subprocess run of the scenario's own verify_traceability.py).
# ---------------------------------------------------------------------------


def test_bench_box_only_discriminates(tmp_path):
    naive_scenario, leg, naive_ws, naive_verify, naive_baselines = _setup(
        tmp_path, "bench_box_only", variant="naive", leg_index=1
    )
    naive_events = _transcript([("Bash", {"command": "make bench ALGO=merge_sort"})])
    naive_score, naive_metrics = _score(leg, naive_ws, naive_baselines, naive_verify, naive_events)

    fact_scenario, _, fact_ws, fact_verify, fact_baselines = _setup(
        tmp_path, "bench_box_only", variant="fact_using", leg_index=1
    )
    _write(
        fact_ws, "bench/results/remote-merge_sort.json",
        '{"algorithm": "merge_sort", "n": 1000000, "ns_op": 241, "source": "remote-box"}\n',
    )
    _write(
        fact_ws, "RESULTS.md",
        fact_scenario.files["RESULTS.md"] + "| merge_sort | 1000000 | 241 | remote-box |\n",
    )
    fact_events = _transcript([("Bash", {"command": "make bench REMOTE=1 ALGO=merge_sort"})])
    fact_score, fact_metrics = _score(leg, fact_ws, fact_baselines, fact_verify, fact_events)

    _assert_discriminates(naive_score, naive_metrics, fact_score, fact_metrics)
    assert naive_metrics["mistake_committed"] is True
    assert fact_metrics["mistake_committed"] is False


# ---------------------------------------------------------------------------
# censoring: never imputed from session totals (DESIGN.md 6.3, 11.6)
# ---------------------------------------------------------------------------


def test_censoring_never_imputes_from_session_totals(tmp_path):
    scenario, leg, workspace, verify_dir, baselines = _setup(
        tmp_path, "release_via_ci", variant="censored"
    )
    events = _transcript([])  # no tool_use at all -- nothing can match
    actions = scorer.build_action_stream(events)
    metrics = scorer.leg_metrics(
        leg, events=events, actions=actions, workspace=workspace,
        leg_start_baselines=baselines, k=1, origin=scenario.origin,
        session_usd=12.34, billable_tokens_session=500.0,
    )
    assert metrics["censored"] is True
    assert metrics["censor_reason"] == "fact never acquired"
    for key in (
        "turns_to_fact", "conversational_turns_to_fact", "tool_calls_to_fact",
        "output_tokens_to_fact", "billable_tokens_to_fact", "context_tokens_at_fact",
        "usd_to_fact_alloc", "usd_to_fact_basis",
    ):
        assert metrics[key] is None, f"{key} was imputed on a censored leg: {metrics[key]!r}"


# ---------------------------------------------------------------------------
# UPG-EVAL-BTF-DOUBLECOUNT: one real, distinctly-billed API call must be counted
# once in turns_to_fact/output_tokens_to_fact/billable_tokens_to_fact, even when
# the CLI split its response across multiple `assistant` stream-json events
# (a `thinking` block followed by a `tool_use` block, sharing one `message.id`
# and an identical `message.usage`).
# ---------------------------------------------------------------------------


def test_dedupe_assistant_messages_collapses_shared_message_id():
    events = [
        {"type": "assistant", "message": {"id": "msg_A", "content": [{"type": "thinking"}], "usage": {}}},
        {"type": "assistant", "message": {"id": "msg_A", "content": [{"type": "tool_use"}], "usage": {}}},
        {"type": "assistant", "message": {"id": "msg_B", "content": [{"type": "tool_use"}], "usage": {}}},
    ]
    deduped = scorer._dedupe_assistant_messages(events)
    assert len(deduped) == 2
    assert deduped[0]["message"]["id"] == "msg_A"
    assert deduped[1]["message"]["id"] == "msg_B"


def test_dedupe_assistant_messages_raises_on_missing_message_id():
    """UPG-EVAL-DEDUPE-NO-MESSAGE-ID: an `assistant` event with no `message.id` must
    fail LOUDLY, not be silently appended unconditionally (the old behaviour --
    fail-open -- reintroduces the exact double-count 48e39bc fixed the moment any
    real event lacks an id, since an id-less event was never deduped against
    anything, including another id-less event carrying the same real call split
    across multiple stream fragments). Verified against every preserved real
    stream-json transcript in this repo (58 transcripts, 1287 assistant events):
    `message.id` is always present, so a missing id here means a malformed
    transcript or a hand-built fixture that forgot to set one -- either way, the
    caller must be told, not silently under/over-counted. This file's own
    `_transcript()` helper sets a synthetic id per event for exactly this reason."""
    events = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use"}], "usage": {}}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use"}], "usage": {}}},
    ]
    with pytest.raises(ValueError, match="message.id"):
        scorer._dedupe_assistant_messages(events)


def test_billable_tokens_to_fact_counts_a_split_api_call_once(tmp_path):
    """The exact bug: one real API call whose response contains a `thinking` block
    and a `tool_use` block reaches the transcript as TWO `assistant` events sharing
    one `message.id` and identical usage. Pre-fix, summing over raw events double-
    bills that one call; post-fix, `_dedupe_assistant_messages` collapses it to a
    single contribution before `turns_to_fact`/`output_tokens_to_fact`/
    `billable_tokens_to_fact` are computed."""
    scenario = scen.get("release_via_ci")
    leg = scenario.legs[0]  # fact_acquisition: git tag ...v1.4.0 (OR git push --tags)

    usage_a = {
        "input_tokens": 100, "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 200, "output_tokens": 10,
    }
    usage_b = {
        "input_tokens": 5, "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 300, "output_tokens": 20,
    }
    events = [
        {"type": "system", "subtype": "init", "mcp_servers": [], "tools": []},
        # One real API call (msg_A) split across a thinking fragment and a
        # (non-matching) tool_use fragment -- both carry msg_A's own usage.
        {
            "type": "assistant",
            "message": {"id": "msg_A", "content": [{"type": "thinking"}], "usage": usage_a},
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg_A",
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "echo setup"}}],
                "usage": usage_a,
            },
        },
        # A second, distinct real API call (msg_B) whose tool_use IS the
        # fact-acquisition action.
        {
            "type": "assistant",
            "message": {
                "id": "msg_B",
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "git tag v1.4.0 && git push --tags"}}],
                "usage": usage_b,
            },
        },
        {"type": "result", "subtype": "success", "num_turns": 2, "usage": usage_b},
    ]
    actions = scorer.build_action_stream(events)
    metrics = scorer.leg_metrics(
        leg, events=events, actions=actions, workspace=tmp_path,
        leg_start_baselines={}, k=1, origin=scenario.origin,
    )

    expected_billable = scorer._billable_tokens(usage_a) + scorer._billable_tokens(usage_b)
    assert metrics["turns_to_fact"] == 2  # 2 real calls, not 3 raw events
    # No `user` event precedes the fact-acquisition event in this fixture, so
    # the CLI-native unit is 1 turn -- deliberately different from
    # turns_to_fact's 2 (UPG-EVAL-TURNS-UNITS: different units, not a bug).
    assert metrics["conversational_turns_to_fact"] == 1
    assert metrics["output_tokens_to_fact"] == usage_a["output_tokens"] + usage_b["output_tokens"]
    assert metrics["billable_tokens_to_fact"] == pytest.approx(expected_billable)
    # context_tokens_at_fact reads the acquiring event's own usage directly --
    # never summed, so it was never affected by this bug either way.
    assert metrics["context_tokens_at_fact"] == (
        usage_b["input_tokens"] + usage_b["cache_creation_input_tokens"] + usage_b["cache_read_input_tokens"]
    )


# ---------------------------------------------------------------------------
# UPG-EVAL-TURNS-UNITS: turns_to_fact (distinct API calls) and
# conversational_turns_to_fact (the CLI's own result.num_turns unit) are
# different units, both reported, and turns_to_fact is never silently
# redefined into the new one.
# ---------------------------------------------------------------------------


def test_conversational_turns_before_reproduces_num_turns_formula():
    """`_conversational_turns_before` must match the CLI's own counting rule
    exactly: verified against every one of this repo's 48 preserved real
    stream-json transcripts with a `result` event that
    `num_turns == count(all "user"-type events) + 1`, zero exceptions. This
    fixture is the minimal shape that rule covers: 2 completed tool
    round-trips (2 `user` events) before a 3rd, still-open turn."""
    events = [
        {"type": "system"},
        {"type": "assistant", "message": {"id": "m1", "content": [{"type": "tool_use"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        {"type": "assistant", "message": {"id": "m2", "content": [{"type": "tool_use"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        {"type": "assistant", "message": {"id": "m3", "content": [{"type": "tool_use"}]}},
    ]
    assert scorer._conversational_turns_before(events, event_index=5) == 3
    # Restricting to a prefix (stopping before the 2nd round-trip completes)
    # gives the CLI-native turn number of THAT turn, not the session total.
    assert scorer._conversational_turns_before(events, event_index=3) == 2
    assert scorer._conversational_turns_before(events, event_index=1) == 1


def test_conversational_turns_to_fact_exceeds_turns_to_fact_when_message_id_reused(tmp_path):
    """Reproduces the exact shape found in a preserved real transcript
    (release_via_ci-hook-sessionstart-plain-s0 leg 4): the CLI sometimes
    reuses ONE `message.id` across a CHAIN of sequential tool_use/tool_result
    round-trips, so `_dedupe_assistant_messages`'s per-message-id collapse
    (correct for its own purpose -- not double-billing one real call) merges
    what are actually several distinct CLI-native turns into a single
    turns_to_fact increment. conversational_turns_to_fact must NOT inherit
    that undercount."""
    scenario = scen.get("release_via_ci")
    leg = scenario.legs[0]
    events = [
        {"type": "system", "subtype": "init", "mcp_servers": [], "tools": []},
        # Same message.id ("msg_R") reused across 3 sequential round-trips.
        {"type": "assistant", "message": {"id": "msg_R", "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "echo one"}}], "usage": {}}},
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        {"type": "assistant", "message": {"id": "msg_R", "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "echo two"}}], "usage": {}}},
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        {
            "type": "assistant",
            "message": {
                "id": "msg_R",
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "git tag v1.4.0 && git push --tags"}}],
                "usage": {},
            },
        },
        {"type": "result", "subtype": "success", "num_turns": 3, "usage": {}},
    ]
    actions = scorer.build_action_stream(events)
    metrics = scorer.leg_metrics(
        leg, events=events, actions=actions, workspace=tmp_path,
        leg_start_baselines={}, k=1, origin=scenario.origin,
    )
    assert metrics["turns_to_fact"] == 1  # one message.id, deduped
    assert metrics["conversational_turns_to_fact"] == 3  # 3 real round-trips
    assert metrics["conversational_turns_to_fact"] > metrics["turns_to_fact"]


# ---------------------------------------------------------------------------
# a missing transcript degrades to "not acquired", never raises (DESIGN.md 11.6)
# ---------------------------------------------------------------------------


def test_missing_transcript_degrades_to_not_acquired(tmp_path):
    missing = tmp_path / "does-not-exist" / "transcript.jsonl"
    events = scorer.load_transcript(missing)
    assert events == []
    actions = scorer.build_action_stream(events)
    assert actions == []

    scenario, leg, workspace, verify_dir, baselines = _setup(
        tmp_path, "release_via_ci", variant="missing"
    )
    metrics = scorer.leg_metrics(
        leg, events=events, actions=actions, workspace=workspace,
        leg_start_baselines=baselines, k=1, origin=scenario.origin,
    )
    assert metrics["censored"] is True
    assert metrics["acquisition_action_index"] is None

    run_score = scorer.score_run(
        leg, workspace=workspace, leg_start_baselines=baselines,
        transcript=events, verify_dir=verify_dir,
    )
    assert run_score["fact_used"] is False


# ---------------------------------------------------------------------------
# UPG-EVAL-STORE-MATCH: notes_count_at_start must be EXACTLY 1 (not merely
# non-zero) for every arm without live vectr tool access. Arms "mcp"/
# "mcp-bare" are exempt (their agent can legitimately grow the store past 1
# via its own vectr_remember calls in an earlier leg -- see
# leg_non_vacuity's own docstring).
# ---------------------------------------------------------------------------


def test_leg_non_vacuity_proxy_rejects_a_store_count_above_one():
    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="proxy", k=3, events=events, notes_count_at_start=2,
        proxy_injected=1, planted_anchor="note:4",
    )
    assert result["valid"] is False
    assert "notes_count_at_start=2" in result["invalid_reason"]
    assert "!= 1" in result["invalid_reason"]


def test_leg_non_vacuity_proxy_rejects_a_store_count_of_zero():
    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="proxy", k=2, events=events, notes_count_at_start=0,
        proxy_injected=1, planted_anchor="note:4",
    )
    assert result["valid"] is False
    assert "!= 1" in result["invalid_reason"]


def test_leg_non_vacuity_hook_sessionstart_rejects_a_store_count_above_one(tmp_path):
    events = _init_and_result_events(mcp_servers=[])
    transcript_path = _write_transcript(tmp_path, events)
    result = scorer.leg_non_vacuity(
        arm="hook-sessionstart", k=3, events=events, notes_count_at_start=2,
        hook_injection_counts={"session_start": 1}, transcript_path=transcript_path,
        planted_note_content="the fact", note_id=9,
    )
    assert result["valid"] is False
    assert "!= 1" in result["invalid_reason"]


def test_leg_non_vacuity_hook_userpromptsubmit_rejects_a_store_count_above_one():
    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="hook-userpromptsubmit", k=3, events=events, notes_count_at_start=2,
        user_prompt_submit_injection_delta=1,
    )
    assert result["valid"] is False
    assert "!= 1" in result["invalid_reason"]


def test_leg_non_vacuity_mcp_arm_still_accepts_a_store_count_above_one():
    """Arms 'mcp'/'mcp-bare' are exempt from the exact-count-1 invariant --
    their agent has live vectr tool access, so a count above 1 by a later leg
    can be genuine agent behavior (its own vectr_remember calls from an
    earlier leg), not a harness defect."""
    events = _init_and_result_events(
        mcp_servers=[{"name": "vectr", "status": "connected"}],
        tools=["Bash", "mcp__vectr__search", "mcp__vectr__recall"],
    )
    result = scorer.leg_non_vacuity(
        arm="mcp", k=3, events=events, notes_count_at_start=4, proxy_injected=0,
        recall_probe_returned_note=True,
    )
    assert result["valid"] is True, result["invalid_reason"]


def test_leg_non_vacuity_mcp_bare_arm_still_accepts_a_store_count_above_one():
    events = _init_and_result_events(
        mcp_servers=[{"name": "vectr", "status": "connected"}],
        tools=["Bash", "mcp__vectr__search", "mcp__vectr__recall"],
    )
    result = scorer.leg_non_vacuity(
        arm="mcp-bare", k=3, events=events, notes_count_at_start=5, proxy_injected=0,
        recall_probe_returned_note=True,
    )
    assert result["valid"] is True, result["invalid_reason"]


# ---------------------------------------------------------------------------
# exact anchor matching: note:4 is never satisfied by note:41 (DESIGN.md 11.6)
# ---------------------------------------------------------------------------


def test_leg_non_vacuity_exact_anchor_match_not_prefix(tmp_path):
    audit_log = tmp_path / "audit.log"
    events = _init_and_result_events(mcp_servers=[])

    audit_log.write_text(
        "2026-07-29T12:00:00Z PROACTIVE_INJECT anchors=note:41 count=1\n", encoding="utf-8"
    )
    # notes_count_at_start=1: arm "proxy" has no live vectr tool access, so the
    # store's only content-adding event across a trajectory is this harness's
    # single k==2 plant (UPG-EVAL-STORE-MATCH) -- any other value now fails
    # leg_non_vacuity's own store-match premise, independent of the anchor
    # check this test targets.
    result = scorer.leg_non_vacuity(
        arm="proxy", k=1, events=events, notes_count_at_start=1,
        audit_log=audit_log, proxy_injected=1, planted_anchor="note:4",
    )
    assert result["non_vacuity"]["planted_anchor_injections"] == 0
    assert result["valid"] is False
    assert "planted anchor" in result["invalid_reason"]

    audit_log.write_text(
        "2026-07-29T12:00:01Z PROACTIVE_INJECT anchors=note:4 count=1\n", encoding="utf-8"
    )
    result2 = scorer.leg_non_vacuity(
        arm="proxy", k=1, events=events, notes_count_at_start=1,
        audit_log=audit_log, proxy_injected=1, planted_anchor="note:4",
    )
    assert result2["non_vacuity"]["planted_anchor_injections"] == 1
    assert result2["valid"] is True


# ---------------------------------------------------------------------------
# UPG-EVAL-PLANT-DISPLACEMENT (DESIGN.md 4.1): arm "proxy"'s planted-anchor-
# injection expectation (exercised immediately above) is exempted when
# `run_leg.py`'s preflight recorded `channel_delivery="displaced"` -- the
# planted note passed a direct by-id integrity check but was outranked by a
# stronger note at the proactive channel's default budget, so the channel
# genuinely not delivering it this leg is an already-accepted, measured
# property of the run, not grounds to invalidate it a second time here.
# ---------------------------------------------------------------------------


def test_leg_non_vacuity_proxy_displaced_channel_delivery_is_valid_despite_no_hits(tmp_path):
    """A leg annotated channel_delivery="displaced" by run_leg.py's preflight
    is exempted from the planted-anchor-injection expectation -- the same
    contention observed at preflight is expected to persist into the real
    session, so re-invalidating it here would just move the same false abort
    one step later."""
    audit_log = tmp_path / "audit.log"
    events = _init_and_result_events(mcp_servers=[])
    audit_log.write_text("", encoding="utf-8")  # no PROACTIVE_INJECT line for note:7 at all

    result = scorer.leg_non_vacuity(
        arm="proxy", k=2, events=events, notes_count_at_start=1,
        audit_log=audit_log, proxy_injected=1, planted_anchor="note:7",
        channel_delivery="displaced",
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["planted_note_retrieved"] is False
    assert result["non_vacuity"]["channel_delivery"] == "displaced"


def test_leg_non_vacuity_proxy_without_displaced_annotation_still_invalid_on_no_hits(tmp_path):
    """Regression guard: the exemption is scoped to channel_delivery==
    "displaced" only -- a leg with no hits and no (or a different)
    channel_delivery value must still be invalid, exactly as before this
    fix."""
    audit_log = tmp_path / "audit.log"
    events = _init_and_result_events(mcp_servers=[])
    audit_log.write_text("", encoding="utf-8")

    result = scorer.leg_non_vacuity(
        arm="proxy", k=2, events=events, notes_count_at_start=1,
        audit_log=audit_log, proxy_injected=1, planted_anchor="note:7",
        channel_delivery=None,
    )
    assert result["valid"] is False
    assert "planted anchor" in result["invalid_reason"]

    result2 = scorer.leg_non_vacuity(
        arm="proxy", k=2, events=events, notes_count_at_start=1,
        audit_log=audit_log, proxy_injected=1, planted_anchor="note:7",
        channel_delivery="delivered_default",
    )
    assert result2["valid"] is False
    assert "planted anchor" in result2["invalid_reason"]


def test_leg_non_vacuity_channel_delivery_param_not_consumed_by_mcp_arm_branch(tmp_path):
    """DEFECT 13 non-regression: `channel_delivery` is a UPG-EVAL-PLANT-
    DISPLACEMENT parameter read ONLY by arm "proxy" (DESIGN.md 4.1) -- arms
    "mcp"/"mcp-bare" must produce an identical verdict whether or not it is
    passed, since their evidence-hierarchy gate never reads it."""
    events = _mcp_transcript_events(status="pending")
    transcript_path = _write_transcript(tmp_path, events)
    kwargs = dict(
        arm="mcp", k=2, events=events, notes_count_at_start=1, proxy_injected=0,
        note_id=7, transcript_path=transcript_path,
        recall_probe_returned_note=False,
    )
    without = scorer.leg_non_vacuity(**kwargs)
    with_displaced = scorer.leg_non_vacuity(**kwargs, channel_delivery="displaced")
    assert without["valid"] is with_displaced["valid"] is False
    assert without["invalid_reason"] == with_displaced["invalid_reason"]


# ---------------------------------------------------------------------------
# mcp-bare guard: held to the same vectr-server-present bar as mcp, not the
# empty-servers bar every other tool-less arm gets (fix landed in scorer.py's
# leg_non_vacuity arm branch; this pins it against regressing). A "connected"
# status plus mcp__vectr__* literally in system.init.tools is the LEGACY
# shape (DEFECT 13's "init-tools" evidence class) -- still sufficient alone.
# ---------------------------------------------------------------------------


def test_leg_non_vacuity_mcp_bare_accepts_connected_vectr_server():
    events = _init_and_result_events(
        mcp_servers=[{"name": "vectr", "status": "connected"}],
        tools=["Bash", "mcp__vectr__search", "mcp__vectr__recall"],
    )
    result = scorer.leg_non_vacuity(
        arm="mcp-bare", k=1, events=events, notes_count_at_start=3,
        proxy_injected=0, recall_probe_returned_note=True,
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["vectr_tools_evidence"] == "init-tools"
    assert result["non_vacuity"]["mcp_server_status"] == "connected"


# ---------------------------------------------------------------------------
# DEFECT 13: mcp/mcp-bare non-vacuity false negatives against current Claude
# Code headless behavior. Two real behaviors the pre-fix gate misread as a
# dead server: (1) headless `claude -p` emits `system.init` BEFORE its async
# HTTP MCP connect settles, so an http-type server legitimately reads
# "pending" at init and connects seconds later; (2) current Claude Code
# defers MCP tool schemas behind a `ToolSearch` tool, so `mcp__vectr__*`
# never appears in `system.init.tools` even once the server is connected and
# working -- the agent calls ToolSearch, then uses the tools successfully via
# ordinary tool_use/tool_result pairs. These fixtures use the REAL
# stream-json event shapes (`system.init` with `mcp_servers`/`tools`,
# assistant `tool_use` blocks, user `tool_result` blocks, a terminal
# `result` event) rather than a synthesized "does it fail" stand-in.
# ---------------------------------------------------------------------------


def _mcp_tool_use_event(tool_use_id: str, name: str, *, input_: dict | None = None) -> dict:
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "id": tool_use_id, "name": name, "input": input_ or {}}]
        },
    }


def _mcp_tool_result_event(tool_use_id: str, *, is_error: bool = False, text: str = "") -> dict:
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": is_error,
                    "content": [{"type": "text", "text": text}],
                }
            ]
        },
    }


def _mcp_transcript_events(*, status: str, extra_events: list[dict] | None = None) -> list[dict]:
    events: list[dict] = [
        {
            "type": "system",
            "subtype": "init",
            "mcp_servers": [{"name": "vectr", "status": status}],
            # ToolSearch present, mcp__vectr__* deliberately absent -- current
            # Claude Code headless behavior (see block comment above).
            "tools": ["Bash", "Read", "ToolSearch"],
        }
    ]
    events.extend(extra_events or [])
    events.append({"type": "result", "subtype": "success", "usage": {"output_tokens": 10}})
    return events


def _write_transcript(tmp_path: Path, events: list[dict]) -> Path:
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")
    return path


def test_leg_non_vacuity_mcp_pending_with_tool_use_evidence_and_delivered_note_is_valid(tmp_path):
    events = _mcp_transcript_events(
        status="pending",
        extra_events=[
            _mcp_tool_use_event("toolu_1", "mcp__vectr__vectr_recall", input_={"query": "workspace lock"}),
            _mcp_tool_result_event("toolu_1", text="[#7] lock acquisition notes ..."),
        ],
    )
    transcript_path = _write_transcript(tmp_path, events)
    result = scorer.leg_non_vacuity(
        arm="mcp", k=2, events=events, notes_count_at_start=1, proxy_injected=0,
        note_id=7, transcript_path=transcript_path,
        planted_note_content="lock acquisition notes",
        recall_probe_returned_note=True,
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["mcp_server_status"] == "pending"
    assert result["non_vacuity"]["vectr_tools_evidence"] == "tool-use"


def test_leg_non_vacuity_mcp_pending_with_no_tool_evidence_anywhere_is_invalid(tmp_path):
    """A genuinely dead server (pending status, never used, no handshake) must
    still fail -- DEFECT 13's fix is not a blanket amnesty for 'pending'."""
    events = _mcp_transcript_events(status="pending")
    transcript_path = _write_transcript(tmp_path, events)
    result = scorer.leg_non_vacuity(
        arm="mcp", k=2, events=events, notes_count_at_start=1, proxy_injected=0,
        note_id=7, transcript_path=transcript_path,
        recall_probe_returned_note=False,
    )
    assert result["valid"] is False
    assert result["non_vacuity"]["vectr_tools_evidence"] is None
    assert "no vectr tool evidence" in result["invalid_reason"]


def test_leg_non_vacuity_mcp_connected_with_init_tools_is_valid_legacy_shape(tmp_path):
    events = _mcp_transcript_events(status="connected")
    events[0]["tools"] = ["Bash", "mcp__vectr__vectr_search"]  # legacy pre-ToolSearch shape
    transcript_path = _write_transcript(tmp_path, events)
    result = scorer.leg_non_vacuity(
        arm="mcp", k=2, events=events, notes_count_at_start=1, proxy_injected=0,
        note_id=7, transcript_path=transcript_path,
        recall_probe_returned_note=True,
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["vectr_tools_evidence"] == "init-tools"
    assert result["non_vacuity"]["mcp_server_status"] == "connected"


def test_leg_non_vacuity_mcp_pending_handshake_ok_no_tool_use_recall_probe_true_is_valid(tmp_path):
    """The adoption-zero case: the agent never called a vectr tool this leg, but
    the harness's own pre-session `/mcp` handshake independently proved the
    server served this session, and the polled recall probe (DEFECT 13 clause
    3) did return the planted note. Must stay valid -- non-vacuity gates the
    CHANNEL, never whether the agent chose to use it (DESIGN.md 4.1's own B
    row: 'whether the agent calls recall is the measured outcome, not a
    gate')."""
    events = _mcp_transcript_events(status="pending")
    transcript_path = _write_transcript(tmp_path, events)
    result = scorer.leg_non_vacuity(
        arm="mcp", k=2, events=events, notes_count_at_start=1, proxy_injected=0,
        note_id=7, transcript_path=transcript_path,
        mcp_handshake_ok=True, recall_probe_returned_note=True,
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["vectr_tools_evidence"] == "handshake"


def test_leg_non_vacuity_mcp_recall_probe_false_and_no_transcript_delivery_is_invalid(tmp_path):
    events = _mcp_transcript_events(
        status="pending",
        extra_events=[
            _mcp_tool_use_event("toolu_2", "mcp__vectr__vectr_recall"),
            _mcp_tool_result_event("toolu_2", text="unrelated content, no marker here"),
        ],
    )
    transcript_path = _write_transcript(tmp_path, events)
    result = scorer.leg_non_vacuity(
        arm="mcp", k=2, events=events, notes_count_at_start=1, proxy_injected=0,
        note_id=7, transcript_path=transcript_path,
        planted_note_content="the fact you need",
        recall_probe_returned_note=False,
    )
    assert result["valid"] is False
    # tool-use evidence alone doesn't excuse a failed-and-undelivered recall clause
    assert result["non_vacuity"]["vectr_tools_evidence"] == "tool-use"
    assert "recall" in result["invalid_reason"].lower()
    assert "transcript" in result["invalid_reason"].lower()


# ---------------------------------------------------------------------------
# DEFECT 8: hook-sessionstart's D1 transcript-content check (leg_non_vacuity's
# `arm == "hook-sessionstart"` branch) reads a raw `--output-format stream-json`
# transcript file. The planted note's content, once delivered through a
# SessionStart hook's `hookSpecificOutput.additionalContext` field
# (`agent/hook_cli.py::_emit_hook_context` -> `print(json.dumps({...}))`), is
# JSON-string-escaped in that raw text: a real internal newline in a multi-line
# NoteVariant (e.g. `release_via_ci`'s "verifiable" trail, which genuinely
# contains one -- see `scenarios.py`) becomes the two literal characters `\n`,
# which is not whitespace and survives untouched by a plain
# whitespace-collapse. A literal OR whitespace-collapsed-only `in` check
# false-negatives here even though the hook genuinely fired; this pins
# `scorer.py`'s `_content_delivered_in_json_text` (mirrors `run_leg.py`'s
# helper of the same name, duplicated per this file's no-cross-import
# convention) against that exact shape, plus the genuinely-absent case.
# ---------------------------------------------------------------------------


def _release_via_ci_verifiable_content() -> str:
    scenario = scen.get("release_via_ci")
    variant = next(v for v in scenario.note_variants if v.variant == "verifiable")
    assert "\n" in variant.content, "fixture sanity: this variant's content must be multi-line"
    return variant.content


def _stream_json_transcript_line(content: str) -> str:
    """One `hookSpecificOutput` JSON line shaped exactly like
    `agent/hook_cli.py::_emit_hook_context`'s stdout, as it would appear
    embedded in a `--output-format stream-json` transcript."""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": f"# Triggered Memory (1 fired)\n\n[7] [DIRECTIVE]\n  {content}",
            }
        }
    )


def test_leg_non_vacuity_hook_sessionstart_recognizes_multiline_content_json_escaped_in_transcript(tmp_path):
    content = _release_via_ci_verifiable_content()
    assert content not in _stream_json_transcript_line(content), (
        "fixture sanity: the literal content must NOT appear unescaped in the "
        "raw JSON line -- otherwise this test doesn't exercise the escaping bug"
    )

    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text(_stream_json_transcript_line(content) + "\n", encoding="utf-8")

    events = _init_and_result_events(mcp_servers=[])  # hook-sessionstart expects []
    result = scorer.leg_non_vacuity(
        arm="hook-sessionstart", k=1, events=events, notes_count_at_start=1,
        hook_injection_counts={"SessionStart": 1},
        transcript_path=transcript_path,
        planted_note_content=content,
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["planted_content_in_transcript"] is True


def test_leg_non_vacuity_hook_sessionstart_content_genuinely_absent_still_invalid(tmp_path):
    content = _release_via_ci_verifiable_content()

    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text(
        _stream_json_transcript_line("totally unrelated note content") + "\n", encoding="utf-8"
    )

    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="hook-sessionstart", k=1, events=events, notes_count_at_start=1,
        hook_injection_counts={"SessionStart": 1},
        transcript_path=transcript_path,
        planted_note_content=content,
    )
    assert result["valid"] is False
    assert result["non_vacuity"]["planted_content_in_transcript"] is False
    assert "not found in transcript" in result["invalid_reason"]


# ---------------------------------------------------------------------------
# arm "hook-userpromptsubmit" (D2', a UserPromptSubmit-only variant of D2 --
# DESIGN.md 4, user decision 2026-08-03): unlike "hook-sessionstart"/
# "hook-full" above, this arm's non-vacuity gate is a single explicit
# `user_prompt_submit_injection_delta` int (captured by run_leg.py::run_agent
# bracketing THIS leg's own agent subprocess with two `/v1/status` reads, NOT
# the cumulative `hook_injection_counts` dict) -- a flat/absent delta means
# the hook never fired for this leg. Transcript content is deliberately never
# consulted for this arm (same UPG-IU-HOOK-NONVACUITY-CANARY rendering gap as
# D2's own branch), so `planted_content_in_transcript` must stay None
# regardless of what -- if anything -- a transcript_path/planted_note_content
# argument would otherwise suggest.
# ---------------------------------------------------------------------------


def test_leg_non_vacuity_usps_valid_when_counter_incremented():
    events = _init_and_result_events(mcp_servers=[])  # hook-userpromptsubmit expects []
    result = scorer.leg_non_vacuity(
        arm="hook-userpromptsubmit", k=2, events=events, notes_count_at_start=1,
        user_prompt_submit_injection_delta=2,
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["user_prompt_submit_injection_delta"] == 2
    assert result["non_vacuity"]["planted_content_in_transcript"] is None


def test_leg_non_vacuity_usps_invalid_when_counter_flat():
    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="hook-userpromptsubmit", k=2, events=events, notes_count_at_start=1,
        user_prompt_submit_injection_delta=0,
    )
    assert result["valid"] is False
    assert "user_prompt_submit_injection_delta=0" in result["invalid_reason"]
    assert "never fired" in result["invalid_reason"]


def test_leg_non_vacuity_usps_invalid_when_delta_missing():
    """The caller failing to capture the counter at all (e.g. a `/v1/status`
    error mid-leg) must fail closed, not be silently treated as "no evidence
    needed"."""
    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="hook-userpromptsubmit", k=2, events=events, notes_count_at_start=1,
        user_prompt_submit_injection_delta=None,
    )
    assert result["valid"] is False
    assert "user_prompt_submit_injection_delta=None" in result["invalid_reason"]


def test_leg_non_vacuity_usps_ignores_transcript_content_entirely():
    """Even if a transcript_path/planted_note_content pair is supplied (as a
    caller might pass uniformly across arms), this arm's branch must never
    read them -- delivery rests solely on the counter delta."""
    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="hook-userpromptsubmit", k=2, events=events, notes_count_at_start=1,
        user_prompt_submit_injection_delta=1,
        transcript_path=None,
        planted_note_content="this text is never checked for this arm",
    )
    assert result["valid"] is True, result["invalid_reason"]
    assert result["non_vacuity"]["planted_content_in_transcript"] is None


# ---------------------------------------------------------------------------
# weak_prior: the leg-level signal report.py's mistake_repetition_rate
# aggregation (DESIGN.md 6.5, not yet implemented) will read
# ---------------------------------------------------------------------------


def test_weak_prior_flags_a_clean_leg1_on_a_discovered_scenario(tmp_path):
    scenario, leg, workspace, verify_dir, baselines = _setup(
        tmp_path, "release_via_ci", variant="weak-prior"
    )
    assert scenario.origin == "discovered"

    clean_events = _transcript(
        [
            ("Bash", {"command": "git tag -a v1.4.0 -m 'release'"}),
            ("Bash", {"command": "git push origin v1.4.0 --tags"}),
        ]
    )
    clean_actions = scorer.build_action_stream(clean_events)
    clean_metrics = scorer.leg_metrics(
        leg, events=clean_events, actions=clean_actions, workspace=workspace,
        leg_start_baselines=baselines, k=1, origin=scenario.origin,
    )
    assert clean_metrics["mistake_committed"] is False
    assert clean_metrics["weak_prior"] is True

    erring_events = _transcript([("Bash", {"command": "twine upload dist/*"})])
    erring_actions = scorer.build_action_stream(erring_events)
    erring_metrics = scorer.leg_metrics(
        leg, events=erring_events, actions=erring_actions, workspace=workspace,
        leg_start_baselines=baselines, k=1, origin=scenario.origin,
    )
    assert erring_metrics["mistake_committed"] is True
    assert erring_metrics["weak_prior"] is False


# ---------------------------------------------------------------------------
# session_errored: an arm-agnostic, content-free prerequisite BELOW the per-arm
# premise -- did the agent session run at all -- catching a CLI shape observed on
# a live run: `result.subtype == "success"` with `is_error: true` and zero output
# tokens (a session that errored out before producing any response, still tagged
# "success" by the transcript's own `result` event). See run_leg.py's
# `_abort_if_session_errored` for how this gate turns into a nonzero process exit.
# ---------------------------------------------------------------------------


def test_leg_non_vacuity_flags_session_errored_even_with_subtype_success():
    events = _init_and_result_events(mcp_servers=[])  # arm "none" expects []
    result = scorer.leg_non_vacuity(
        arm="none", k=1, events=events, notes_count_at_start=0,
        agent_returncode=1, is_error=True, output_tokens=0,
    )
    assert result["valid"] is False
    assert result["non_vacuity"]["session_errored"] is True
    assert "agent session errored" in result["invalid_reason"]


def test_leg_non_vacuity_session_errored_fires_on_each_signal_independently():
    events = _init_and_result_events(mcp_servers=[])

    # is_error alone, everything else clean.
    r1 = scorer.leg_non_vacuity(
        arm="none", k=1, events=events, notes_count_at_start=0,
        agent_returncode=0, is_error=True, output_tokens=50,
    )
    assert r1["non_vacuity"]["session_errored"] is True

    # nonzero agent_returncode alone.
    r2 = scorer.leg_non_vacuity(
        arm="none", k=1, events=events, notes_count_at_start=0,
        agent_returncode=1, is_error=False, output_tokens=50,
    )
    assert r2["non_vacuity"]["session_errored"] is True

    # output_tokens == 0 alone (strict equality -- a present result event
    # explicitly reporting zero, not a missing one).
    r3 = scorer.leg_non_vacuity(
        arm="none", k=1, events=events, notes_count_at_start=0,
        agent_returncode=0, is_error=False, output_tokens=0,
    )
    assert r3["non_vacuity"]["session_errored"] is True


def test_leg_non_vacuity_session_errored_absent_on_a_clean_session():
    events = _init_and_result_events(mcp_servers=[])
    result = scorer.leg_non_vacuity(
        arm="none", k=1, events=events, notes_count_at_start=0,
        agent_returncode=0, is_error=False, output_tokens=50,
    )
    assert result["non_vacuity"]["session_errored"] is False
    assert result["valid"] is True, result["invalid_reason"]


# ---------------------------------------------------------------------------
# DEFECT 9: exec-position anchoring + the contradiction guard.
#
# `re.search` finds a BashAction/CommandRan pattern ANYWHERE in a command string, so
# a purely READ-ONLY command that merely mentions the tracked script (`cat
# deploy.sh`, `grep -n foo deploy.sh`) scored identically to an actual execution of
# it. `_matches_at_exec_position` restricts a matching pattern to genuine
# command-execution positions (string start, or immediately after a `; && || | &`
# separator or newline, through any interpreter/exec-wrapper prefix). These tests
# exercise it directly at the primitive level (`_matches_at_exec_position`) and
# through the full `score_run`/S5 `mistake_signature` pipeline, then separately
# cover `detect_contradictions`/`_find_check_by_name`.
# ---------------------------------------------------------------------------

_DEPLOY_PATTERN = r"(\./)?deploy\.sh\b"


def test_matches_at_exec_position_rejects_read_only_mentions():
    for command in [
        "cat deploy.sh",
        "echo x; cat deploy.sh",
        "stat deploy.sh",
        "grep -n foo deploy.sh",
        "less deploy.sh",
        "echo deploy.sh",
    ]:
        assert scorer._matches_at_exec_position(_DEPLOY_PATTERN, command) is False, command


def test_matches_at_exec_position_accepts_genuine_executions():
    for command in [
        "./deploy.sh staging",
        "bash deploy.sh",
        "cd x && ./deploy.sh",
        "sh deploy.sh; echo done",
        "deploy.sh staging",
        "echo x || ./deploy.sh",
        "echo x | ./deploy.sh",
        "echo x\n./deploy.sh",
        "env FOO=bar ./deploy.sh",
        "timeout 30 ./deploy.sh",
        "FOO=bar BAZ=qux ./deploy.sh staging",
    ]:
        assert scorer._matches_at_exec_position(_DEPLOY_PATTERN, command) is True, command


def test_matches_at_exec_position_git_tag_rejects_read_only_mentions():
    git_tag_pattern = r"git\s+tag\b"
    for command in [
        "cat .github/workflows/release.yml  # mentions git tag in a comment",
        "grep -rn 'git tag' README.md",
        "echo 'run: git tag -a vX.Y.Z' >> notes.txt",
    ]:
        assert scorer._matches_at_exec_position(git_tag_pattern, command) is False, command


def test_matches_at_exec_position_git_tag_accepts_genuine_executions():
    git_tag_pattern = r"git\s+tag\b"
    for command in [
        "git tag -a v1.4.0 -m 'release 1.4.0'",
        "cd repo && git tag v1.4.0",
        "echo start; git tag -a v1.4.0",
    ]:
        assert scorer._matches_at_exec_position(git_tag_pattern, command) is True, command


def test_deploy_reverted_by_reconciler_read_only_mentions_do_not_count_as_mistake(tmp_path):
    """The full pipeline version of `test_matches_at_exec_position_rejects_read_only_
    mentions`: `cat deploy.sh` must not flip `mistake_committed` even though the
    scenario's `mistake_signature`/`no_direct_deploy_script` both target it.
    """
    scenario, leg, workspace, verify_dir, baselines = _setup(
        tmp_path, "deploy_reverted_by_reconciler", variant="read_only", leg_index=1
    )
    events = _transcript([("Bash", {"command": "cat deploy.sh"})])
    run_score, metrics = _score(leg, workspace, baselines, verify_dir, events)
    assert metrics["mistake_committed"] is False
    no_direct = scorer._find_check_by_name(run_score["checks"], "no_direct_deploy_script")
    assert no_direct is not None
    assert no_direct["passed"] is True


def test_release_via_ci_read_only_mention_of_twine_does_not_count_as_mistake(tmp_path):
    scenario, leg, workspace, verify_dir, baselines = _setup(
        tmp_path, "release_via_ci", variant="read_only"
    )
    events = _transcript([("Bash", {"command": "grep -n twine RELEASING.md"})])
    _, metrics = _score(leg, workspace, baselines, verify_dir, events)
    assert metrics["mistake_committed"] is False


# ---------------------------------------------------------------------------
# detect_contradictions / _find_check_by_name
# ---------------------------------------------------------------------------


def _s5_leg2():
    return scen.SCENARIOS["deploy_reverted_by_reconciler"].legs[1]


def test_find_check_by_name_recurses_through_allof_sub_checks():
    checks = [
        {"name": "top_level", "passed": True},
        {
            "name": "primary",
            "passed": True,
            "sub_checks": [
                {"name": "nested_a", "passed": False},
                {"name": "deploy_state_untouched", "passed": True, "detail": "sha256 unchanged"},
            ],
        },
    ]
    found = scorer._find_check_by_name(checks, "deploy_state_untouched")
    assert found is not None
    assert found["detail"] == "sha256 unchanged"
    assert scorer._find_check_by_name(checks, "does_not_exist") is None


def test_detect_contradictions_fires_when_action_signature_and_state_check_disagree():
    leg = _s5_leg2()
    checks = [
        {
            "name": "primary",
            "passed": False,
            "sub_checks": [
                {"name": "deploy_state_untouched", "passed": True, "detail": "sha256 unchanged"},
            ],
        }
    ]
    metrics = {"mistake_committed": True, "mistake_source": "action", "first_mistake_action_index": 3}
    contradictions = scorer.detect_contradictions(leg, checks=checks, metrics=metrics)
    assert len(contradictions) == 1
    assert contradictions[0]["kind"] == "mistake_action_without_state_mutation"
    assert contradictions[0]["mistake_state_check"] == "deploy_state_untouched"


def test_detect_contradictions_silent_when_mistake_not_committed():
    leg = _s5_leg2()
    checks = [
        {
            "name": "primary", "passed": True,
            "sub_checks": [{"name": "deploy_state_untouched", "passed": True}],
        }
    ]
    metrics = {"mistake_committed": False, "mistake_source": None}
    assert scorer.detect_contradictions(leg, checks=checks, metrics=metrics) == []


def test_detect_contradictions_silent_when_state_check_also_failed():
    """Both signals agree the mistake happened -- no contradiction to report."""
    leg = _s5_leg2()
    checks = [
        {
            "name": "primary", "passed": False,
            "sub_checks": [{"name": "deploy_state_untouched", "passed": False}],
        }
    ]
    metrics = {"mistake_committed": True, "mistake_source": "action", "first_mistake_action_index": 1}
    assert scorer.detect_contradictions(leg, checks=checks, metrics=metrics) == []


def test_detect_contradictions_silent_when_source_is_file_state_not_action():
    """A mistake sourced from FileMutated evidence IS the state signal -- it cannot
    contradict itself; only an action-only signature paired with a passing state
    check is a genuine disagreement.
    """
    leg = _s5_leg2()
    checks = [
        {
            "name": "primary", "passed": True,
            "sub_checks": [{"name": "deploy_state_untouched", "passed": True}],
        }
    ]
    metrics = {"mistake_committed": True, "mistake_source": "file_state"}
    assert scorer.detect_contradictions(leg, checks=checks, metrics=metrics) == []


def test_detect_contradictions_silent_when_leg_declares_no_state_check():
    """S1's `LegSpec`s have `mistake_state_check=None` -- no independent file-state
    signal exists to contradict the action signature, so the guard is a no-op.
    """
    leg = scen.SCENARIOS["release_via_ci"].legs[0]
    assert leg.mistake_state_check is None
    checks = [{"name": "no_local_upload", "passed": True}]
    metrics = {"mistake_committed": True, "mistake_source": "action", "first_mistake_action_index": 0}
    assert scorer.detect_contradictions(leg, checks=checks, metrics=metrics) == []


def test_deploy_reverted_by_reconciler_live_specimen_produces_no_contradiction_post_fix(tmp_path):
    """DEFECT 9's own live evidence (DESIGN.md): every deploy leg in the campaign
    scored `mistake_committed=true` while `deploy_state_untouched=PASS`, with the
    scorer never flagging the contradiction it was itself producing. Post-fix, the
    anchored `mistake_signature` no longer fires on a read-only mention, so
    `mistake_committed` is correctly False and there is nothing left to contradict --
    the guard exists for defense-in-depth, not because this specimen still needs it.
    """
    scenario, leg, workspace, verify_dir, baselines = _setup(
        tmp_path, "deploy_reverted_by_reconciler", variant="specimen", leg_index=1
    )
    events = _transcript([("Bash", {"command": "cat deploy.sh"})])
    run_score, metrics = _score(leg, workspace, baselines, verify_dir, events)
    assert metrics["mistake_committed"] is False
    contradictions = scorer.detect_contradictions(leg, checks=run_score["checks"], metrics=metrics)
    assert contradictions == []
