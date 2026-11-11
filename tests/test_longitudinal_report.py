"""Tests for the longitudinal-rediscovery report layer (DESIGN.md section 11, step 5
-- `benchmarks/longitudinal_rediscovery/report.py`).

No live leg has run against this branch (T1 is sentinel-approval-gated; T0 is
probe-only, zero legs), so every test here builds leg records by hand -- the same
fixture style `tests/test_longitudinal_scorer.py` uses for the scorer -- rather than
depending on a real `results.jsonl`.

`mistake_repetition_rate is null when leg 1 is clean` (DESIGN.md 6.5) is a
TRAJECTORY-level aggregate that `tests/test_longitudinal_scorer.py`'s own docstring
explicitly defers to this file (see its lines on `weak_prior` vs this metric); it is
covered below by `test_mistake_repetition_rate_is_null_when_leg1_is_clean` and its
converse `test_mistake_repetition_rate_equals_rate_post_when_leg1_committed_mistake`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "longitudinal_rediscovery"
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
report = _load_local_module("_vectr_eval_longitudinal_report_test", "report.py")


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _metrics(
    *,
    censored: bool = False,
    mistake_committed: bool | None = False,
    self_corrected: bool = False,
    turns_to_fact: int | None = 3,
    tool_calls_to_fact: int | None = 4,
    billable_tokens_to_fact: float | None = 500.0,
    rediscovery_actions: int = 2,
    usd_to_fact_alloc: float | None = 0.05,
) -> dict[str, Any]:
    if censored:
        return {
            "censored": True, "censor_reason": "fact never acquired",
            "mistake_committed": mistake_committed, "self_corrected": self_corrected,
            "turns_to_fact": None, "tool_calls_to_fact": None,
            "billable_tokens_to_fact": None, "rediscovery_actions": rediscovery_actions,
            "usd_to_fact_alloc": None,
        }
    return {
        "censored": False, "censor_reason": None,
        "mistake_committed": mistake_committed, "self_corrected": self_corrected,
        "turns_to_fact": turns_to_fact, "tool_calls_to_fact": tool_calls_to_fact,
        "billable_tokens_to_fact": billable_tokens_to_fact,
        "rediscovery_actions": rediscovery_actions, "usd_to_fact_alloc": usd_to_fact_alloc,
    }


def _leg_record(
    *, scenario: str, arm: str, note_variant: str, seed: int, k: int,
    valid: bool = True, session_usd: float = 0.24, metrics: dict[str, Any] | None = None,
    started_utc: str = "20260101T000000Z", leg_id: str | None = None,
) -> dict[str, Any]:
    return {
        "leg_id": leg_id if leg_id is not None else f"stamp-{scenario}-{arm}-{note_variant}-s{seed}-k{k}",
        "trajectory_id": f"{scenario}-{arm}-{note_variant}-s{seed}",
        "scenario": scenario, "arm": arm, "note_variant": note_variant, "seed": seed, "k": k,
        "valid": valid, "invalid_reason": "" if valid else "primary check failed",
        "started_utc": started_utc,
        "metrics": metrics if metrics is not None else _metrics(),
        "cost": {"session_usd": session_usd},
    }


def _leg1_result(*, mistake_committed: bool, censored: bool = False) -> dict[str, Any]:
    return {"metrics": _metrics(censored=censored, mistake_committed=mistake_committed)}


# ---------------------------------------------------------------------------
# rdc_ratio / rdc_delta primitives
# ---------------------------------------------------------------------------


def test_rdc_ratio_null_when_leg1_component_is_zero_or_missing():
    rdc_k = {"turns_to_fact": 2, "tool_calls_to_fact": 3, "billable_tokens_to_fact": 100.0, "rediscovery_actions": 1, "usd_to_fact_alloc": 0.01}
    rdc_1_zero = {c: 0 for c in report.RDC_COMPONENTS}
    rdc_1_missing = {c: None for c in report.RDC_COMPONENTS}
    assert all(v is None for v in report.rdc_ratio(rdc_k, rdc_1_zero).values())
    assert all(v is None for v in report.rdc_ratio(rdc_k, rdc_1_missing).values())


def test_rdc_ratio_divides_componentwise():
    rdc_k = {"turns_to_fact": 2, "tool_calls_to_fact": 3, "billable_tokens_to_fact": 100.0, "rediscovery_actions": 1, "usd_to_fact_alloc": 0.02}
    rdc_1 = {"turns_to_fact": 4, "tool_calls_to_fact": 6, "billable_tokens_to_fact": 400.0, "rediscovery_actions": 2, "usd_to_fact_alloc": 0.04}
    ratios = report.rdc_ratio(rdc_k, rdc_1)
    assert ratios == {"turns_to_fact": 0.5, "tool_calls_to_fact": 0.5, "billable_tokens_to_fact": 0.25, "rediscovery_actions": 0.5, "usd_to_fact_alloc": 0.5}


def test_rdc_delta_is_a_minus_x_and_null_when_either_side_missing():
    rdc_a = {"turns_to_fact": 5, "tool_calls_to_fact": 6, "billable_tokens_to_fact": 500.0, "rediscovery_actions": 3, "usd_to_fact_alloc": 0.08}
    rdc_x = {"turns_to_fact": 2, "tool_calls_to_fact": 2, "billable_tokens_to_fact": None, "rediscovery_actions": 1, "usd_to_fact_alloc": 0.02}
    delta = report.rdc_delta(rdc_a, rdc_x)
    assert delta["turns_to_fact"] == 3
    assert delta["tool_calls_to_fact"] == 4
    assert delta["billable_tokens_to_fact"] is None  # x-side missing -> null, never imputed
    assert delta["rediscovery_actions"] == 2
    assert delta["usd_to_fact_alloc"] == 0.06


# ---------------------------------------------------------------------------
# leg1_rdc_reference (6.4)
# ---------------------------------------------------------------------------


def test_leg1_rdc_reference_told_is_all_zero_regardless_of_leg1_record():
    rdc_1, ref = report.leg1_rdc_reference("told", None)
    assert ref == "told_prompt"
    assert all(v == 0 for v in rdc_1.values())


def test_leg1_rdc_reference_discovered_uses_shared_leg1_metrics():
    leg1_record = {"metrics": _metrics(turns_to_fact=7, tool_calls_to_fact=8, billable_tokens_to_fact=900.0, rediscovery_actions=4, usd_to_fact_alloc=0.09)}
    rdc_1, ref = report.leg1_rdc_reference("discovered", leg1_record)
    assert ref == "shared_leg1"
    assert rdc_1["turns_to_fact"] == 7
    assert rdc_1["usd_to_fact_alloc"] == 0.09


def test_leg1_rdc_reference_missing_is_all_null_not_imputed():
    rdc_1, ref = report.leg1_rdc_reference("discovered", None)
    assert ref == "leg1_missing"
    assert all(v is None for v in rdc_1.values())


def test_leg1_rdc_reference_censored_is_all_null_not_imputed():
    leg1_record = _leg1_result(mistake_committed=False, censored=True)
    rdc_1, ref = report.leg1_rdc_reference("discovered", leg1_record)
    assert ref == "leg1_censored"
    assert all(v is None for v in rdc_1.values())


# ---------------------------------------------------------------------------
# mistake_repetition_rate (6.5) -- the sentinel-mandated test and its converse
# ---------------------------------------------------------------------------


def test_mistake_repetition_rate_is_null_when_leg1_is_clean():
    """DESIGN.md 6.5: mistake_repetition_rate is defined ONLY when leg 1 committed
    the mistake -- null when leg 1 was clean, even though mistake_rate_post (the
    always-defined workhorse) is a real number for the exact same legs. This is the
    expected state for TOLD scenarios (the user has just stated the fact in leg 1).
    """
    legs = [
        _leg_record(scenario="deploy_reverted_by_reconciler", arm="none", note_variant="none", seed=0, k=2, metrics=_metrics(mistake_committed=True)),
        _leg_record(scenario="deploy_reverted_by_reconciler", arm="none", note_variant="none", seed=0, k=3, metrics=_metrics(mistake_committed=True)),
    ]
    leg1_record = _leg1_result(mistake_committed=False)  # leg 1 was clean
    t = report.trajectory_report("deploy_reverted_by_reconciler-none-none-s0", legs, leg1_record, "told")

    assert t["leg1_mistake_committed"] is False
    assert t["mistake_rate_post"] == 1.0  # both k=2 and k=3 repeated it
    assert t["mistake_repetition_rate"] is None  # but there was nothing to "repeat" from leg 1's own state


def test_mistake_repetition_rate_equals_rate_post_when_leg1_committed_mistake():
    legs = [
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, metrics=_metrics(mistake_committed=True)),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=3, metrics=_metrics(mistake_committed=False)),
    ]
    leg1_record = _leg1_result(mistake_committed=True)  # leg 1 DID commit the error
    t = report.trajectory_report("release_via_ci-none-none-s0", legs, leg1_record, "discovered")

    assert t["leg1_mistake_committed"] is True
    assert t["mistake_rate_post"] == 0.5
    assert t["mistake_repetition_rate"] == 0.5


def test_mistake_repetition_rate_is_null_when_leg1_record_is_unavailable():
    legs = [_leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, metrics=_metrics(mistake_committed=True))]
    t = report.trajectory_report("release_via_ci-none-none-s0", legs, None, "discovered")
    assert t["leg1_mistake_committed"] is None
    assert t["mistake_repetition_rate"] is None
    assert t["mistake_rate_post"] == 1.0


def test_censored_leg_still_counts_toward_mistake_metrics():
    """6.4: 'Censored legs still count in full toward the mistake metrics.'"""
    legs = [_leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, metrics=_metrics(censored=True, mistake_committed=True))]
    leg1_record = _leg1_result(mistake_committed=True)
    t = report.trajectory_report("release_via_ci-none-none-s0", legs, leg1_record, "discovered")
    assert t["censored_count"] == 1
    assert t["mistake_rate_post"] == 1.0  # the censored leg's mistake_committed=True is NOT dropped
    assert t["n_legs_counted"] == 1


# ---------------------------------------------------------------------------
# invalid-leg exclusion
# ---------------------------------------------------------------------------


def test_invalid_leg_is_excluded_from_every_number_but_counted():
    legs = [
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, valid=False, metrics=_metrics(mistake_committed=True)),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=3, valid=True, metrics=_metrics(mistake_committed=False)),
    ]
    leg1_record = _leg1_result(mistake_committed=True)
    t = report.trajectory_report("release_via_ci-none-none-s0", legs, leg1_record, "discovered")
    assert t["invalid_count"] == 1
    assert t["n_legs_counted"] == 1  # only the valid k=3 leg counts
    assert t["mistake_rate_post"] == 0.0  # k=3's own mistake_committed=False, k=2 excluded entirely
    row_k2 = next(r for r in t["legs"] if r["k"] == 2)
    assert row_k2 == {"k": 2, "valid": False, "excluded_reason": "invalid"}


# ---------------------------------------------------------------------------
# trajectory identity fields + cross-arm delta
# ---------------------------------------------------------------------------


def test_trajectory_report_carries_identity_fields_from_first_leg():
    legs = [_leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2)]
    t = report.trajectory_report("release_via_ci-proxy-plain-s0", legs, None, "discovered")
    assert (t["scenario"], t["arm"], t["note_variant"], t["seed"]) == ("release_via_ci", "proxy", "plain", 0)


def test_trajectory_report_row_surfaces_context_tokens_at_fact_alongside_rdc():
    """UPG-EVAL-BTF-DOUBLECOUNT: `context_tokens_at_fact` is reported on every
    per-leg row alongside `rdc["billable_tokens_to_fact"]` -- an exact snapshot of
    retained context at the acquiring turn, distinct from the weighted aggregate
    RDC component -- but is never itself an RDC component (rdc_ratio/rdc_delta/
    cross_arm_delta must never see it)."""
    metrics = _metrics()
    metrics["context_tokens_at_fact"] = 4321
    legs = [_leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2, metrics=metrics)]
    t = report.trajectory_report("release_via_ci-proxy-plain-s0", legs, None, "discovered")
    row = t["legs"][0]
    assert row["context_tokens_at_fact"] == 4321
    assert "context_tokens_at_fact" not in row["rdc"]
    assert "context_tokens_at_fact" not in report.RDC_COMPONENTS


def test_trajectory_report_row_context_tokens_at_fact_defaults_to_none_when_absent():
    legs = [_leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, metrics=_metrics())]
    t = report.trajectory_report("release_via_ci-none-none-s0", legs, None, "discovered")
    assert t["legs"][0]["context_tokens_at_fact"] is None


def test_trajectory_report_row_surfaces_conversational_turns_to_fact_and_session_turns_alongside_rdc():
    """UPG-EVAL-TURNS-UNITS: `conversational_turns_to_fact` (the CLI's own
    `result.num_turns` unit, scorer.leg_metrics) and `cost.session_turns` are
    reported on every per-leg row alongside `rdc["turns_to_fact"]` (a distinct-
    API-call count) -- directly comparable to each other, but neither is folded
    into RDC(k) itself, and `turns_to_fact` inside `rdc` keeps its original
    meaning unchanged."""
    metrics = _metrics()
    metrics["conversational_turns_to_fact"] = 7
    legs = [
        _leg_record(
            scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
            metrics=metrics,
        )
    ]
    legs[0]["cost"]["session_turns"] = 9
    t = report.trajectory_report("release_via_ci-proxy-plain-s0", legs, None, "discovered")
    row = t["legs"][0]
    assert row["conversational_turns_to_fact"] == 7
    assert row["session_turns"] == 9
    assert "conversational_turns_to_fact" not in row["rdc"]
    assert "session_turns" not in row["rdc"]
    assert "conversational_turns_to_fact" not in report.RDC_COMPONENTS
    assert "session_turns" not in report.RDC_COMPONENTS
    assert row["rdc"]["turns_to_fact"] == metrics["turns_to_fact"]


def test_trajectory_report_row_conversational_turns_to_fact_and_session_turns_default_to_none_when_absent():
    legs = [_leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, metrics=_metrics())]
    t = report.trajectory_report("release_via_ci-none-none-s0", legs, None, "discovered")
    row = t["legs"][0]
    assert row["conversational_turns_to_fact"] is None
    assert row["session_turns"] is None


def test_cross_arm_delta_only_pairs_matching_k_present_in_both():
    leg1_record = _leg1_result(mistake_committed=True)
    a_legs = [
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, session_usd=0.30, metrics=_metrics(turns_to_fact=6)),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=3, session_usd=0.30, metrics=_metrics(turns_to_fact=6)),
    ]
    x_legs = [
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2, session_usd=0.12, metrics=_metrics(turns_to_fact=2)),
        # no k=3 leg for arm X -- e.g. still running / not yet reached
    ]
    a_report = report.trajectory_report("release_via_ci-none-none-s0", a_legs, leg1_record, "discovered")
    x_report = report.trajectory_report("release_via_ci-proxy-plain-s0", x_legs, leg1_record, "discovered")
    delta = report.cross_arm_delta(a_report, x_report)
    assert len(delta) == 1
    assert delta[0]["k"] == 2
    assert delta[0]["rdc_delta_a_minus_x"]["turns_to_fact"] == 4
    assert delta[0]["session_usd_delta_a_minus_x"] == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# duplicate-line dedup -- results.jsonl is append-only and `run_plan.py`'s
# `_supersede()` only renames the trajectory DIRECTORY on a re-run, never touching
# this file, so a re-run leaves a stale line behind carrying the same
# (trajectory_id, k).
# ---------------------------------------------------------------------------


def test_select_authoritative_legs_picks_latest_started_utc_marks_rest_superseded():
    """Three records share (trajectory_id, k=2): invalid+stale, valid+stale,
    valid+current (mirrors the live evidence: an invalid attempt, a superseded
    valid retry, then the current valid retry). Only the latest-started_utc record
    is authoritative; the other two -- including the invalid one -- are superseded,
    each exactly once.
    """
    invalid_stale = _leg_record(
        scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
        valid=False, started_utc="20260101T210136Z", leg_id="stamp-a-k2",
    )
    valid_stale = _leg_record(
        scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
        valid=True, started_utc="20260101T214428Z", leg_id="stamp-b-k2",
        metrics=_metrics(turns_to_fact=99),
    )
    valid_current = _leg_record(
        scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
        valid=True, started_utc="20260101T223222Z", leg_id="stamp-c-k2",
        metrics=_metrics(turns_to_fact=5),
    )
    k3_only = _leg_record(
        scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=3,
        valid=True, started_utc="20260101T223507Z",
    )
    authoritative, superseded = report.select_authoritative_legs(
        [invalid_stale, valid_stale, valid_current, k3_only]
    )
    assert authoritative == [valid_current, k3_only]
    assert len(superseded) == 2
    assert invalid_stale in superseded
    assert valid_stale in superseded


def test_select_authoritative_legs_ties_break_on_leg_id_deterministically():
    r1 = _leg_record(
        scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
        started_utc="20260101T210000Z", leg_id="aaa-k2",
    )
    r2 = _leg_record(
        scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
        started_utc="20260101T210000Z", leg_id="zzz-k2",
    )
    authoritative, superseded = report.select_authoritative_legs([r1, r2])
    assert authoritative == [r2]  # "zzz" > "aaa" lexicographically -- deterministic, not file order
    assert superseded == [r1]


def test_trajectory_report_surfaces_superseded_rows_without_double_counting_invalid():
    """`trajectory_report` trusts `legs` is already authoritative-only; the extra
    `superseded_legs` for the same trajectory must be surfaced (never silently
    dropped) but never counted toward `invalid_count` or any aggregate -- and the
    stale invalid line among them counts once, as superseded, not also as invalid.
    """
    legs = [
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
                    started_utc="20260101T223222Z", metrics=_metrics(mistake_committed=False, turns_to_fact=5)),
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=3,
                    started_utc="20260101T223507Z", metrics=_metrics(mistake_committed=False)),
    ]
    superseded_legs = [
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
                    valid=False, started_utc="20260101T210136Z", leg_id="stamp-a-k2"),
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
                    valid=True, started_utc="20260101T214428Z", leg_id="stamp-b-k2",
                    metrics=_metrics(turns_to_fact=99)),
    ]
    t = report.trajectory_report(
        "release_via_ci-proxy-plain-s0", legs, None, "discovered", superseded_legs=superseded_legs
    )

    assert t["invalid_count"] == 0  # the stale invalid line is superseded, not invalid
    assert t["superseded_count"] == 2
    assert t["n_legs_counted"] == 2  # k=2 (current) + k=3 -- neither stale duplicate counts

    superseded_rows = [r for r in t["legs"] if r.get("excluded_reason") == "superseded"]
    assert len(superseded_rows) == 2
    assert {r["k"] for r in superseded_rows} == {2}
    counted_row_k2 = next(r for r in t["legs"] if r["k"] == 2 and r.get("excluded_reason") is None)
    assert counted_row_k2["rdc"]["turns_to_fact"] == 5  # the stale 99 never contaminates the counted row


def test_scenario_report_dedups_duplicate_lines_end_to_end(tmp_path):
    """End-to-end reproduction of the live defect: results.jsonl carries three lines
    for one (trajectory, k=2) -- invalid + stale-valid + current-valid -- on the
    proxy arm, AND a stale duplicate at k=2 on the none-arm side too. Aggregates,
    per_leg_rows' counted rows, and cross_arm_delta must all reflect ONLY the
    current line on both sides -- exactly one counted k=2 row and one delta row per
    matched k, not one per duplicate line.
    """
    records = [
        # none arm: stale + current at k=2, current-only k=3
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
                    started_utc="20260101T200000Z", leg_id="none-k2-stale",
                    session_usd=0.99, metrics=_metrics(mistake_committed=True, turns_to_fact=50)),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
                    started_utc="20260101T210000Z", leg_id="none-k2-current",
                    session_usd=0.30, metrics=_metrics(mistake_committed=True, turns_to_fact=6)),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=3,
                    started_utc="20260101T211000Z", leg_id="none-k3",
                    session_usd=0.30, metrics=_metrics(mistake_committed=True, turns_to_fact=6)),
        # proxy arm: invalid-stale + valid-stale + valid-current at k=2, current-only k=3
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
                    valid=False, started_utc="20260101T210136Z", leg_id="proxy-k2-invalid"),
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
                    started_utc="20260101T214428Z", leg_id="proxy-k2-stale",
                    session_usd=0.50, metrics=_metrics(mistake_committed=False, turns_to_fact=99)),
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2,
                    started_utc="20260101T223222Z", leg_id="proxy-k2-current",
                    session_usd=0.12, metrics=_metrics(mistake_committed=False, turns_to_fact=2)),
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=3,
                    started_utc="20260101T223507Z", leg_id="proxy-k3",
                    session_usd=0.12, metrics=_metrics(mistake_committed=False, turns_to_fact=2)),
    ]
    _write_results_jsonl(tmp_path, records)

    sc = report.scenario_report(tmp_path, "release_via_ci")
    none_t = sc["trajectories"]["release_via_ci-none-none-s0"]
    proxy_t = sc["trajectories"]["release_via_ci-proxy-plain-s0"]

    assert none_t["n_legs_counted"] == 2
    assert none_t["superseded_count"] == 1
    assert none_t["invalid_count"] == 0

    assert proxy_t["n_legs_counted"] == 2
    assert proxy_t["superseded_count"] == 2  # invalid-stale AND valid-stale, both non-authoritative
    assert proxy_t["invalid_count"] == 0  # the stale invalid line does not also count as invalid

    counted_k2 = next(r for r in proxy_t["legs"] if r["k"] == 2 and r.get("excluded_reason") is None)
    assert counted_k2["rdc"]["turns_to_fact"] == 2  # the current value, not the stale 99

    delta = sc["cross_arm_delta_vs_none"]["release_via_ci-proxy-plain-s0"]
    assert len(delta) == 2  # exactly one row per matched k (k=2, k=3) -- not one per stale duplicate
    assert {d["k"] for d in delta} == {2, 3}
    delta_k2 = next(d for d in delta if d["k"] == 2)
    assert delta_k2["rdc_delta_a_minus_x"]["turns_to_fact"] == 4  # 6 (current none) - 2 (current proxy)


def test_format_human_readable_marks_superseded_rows_distinctly(tmp_path):
    records = [
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
                    started_utc="20260101T200000Z", leg_id="none-k2-stale",
                    metrics=_metrics(mistake_committed=True)),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
                    started_utc="20260101T210000Z", leg_id="none-k2-current",
                    metrics=_metrics(mistake_committed=True)),
    ]
    _write_results_jsonl(tmp_path, records)
    text = report.format_human_readable(report.build_report(tmp_path, scenarios=["release_via_ci"]))
    assert "SUPERSEDED" in text
    assert "superseded=1" in text


# ---------------------------------------------------------------------------
# scenario_report / build_report -- end-to-end over a real runs-dir layout
# ---------------------------------------------------------------------------


def _write_results_jsonl(runs_dir: Path, records: list[dict]) -> None:
    with (runs_dir / "results.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _write_leg1(runs_dir: Path, scenario: str, seed: int, *, mistake_committed: bool) -> None:
    d = runs_dir / "_shared" / "leg1" / f"{scenario}-s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(_leg1_result(mistake_committed=mistake_committed)))


def test_scenario_report_groups_trajectories_and_computes_cross_arm_delta(tmp_path):
    _write_leg1(tmp_path, "release_via_ci", 0, mistake_committed=True)
    records = [
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, session_usd=0.30, metrics=_metrics(mistake_committed=True, turns_to_fact=6)),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=3, session_usd=0.30, metrics=_metrics(mistake_committed=True, turns_to_fact=6)),
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=2, session_usd=0.12, metrics=_metrics(mistake_committed=False, turns_to_fact=2)),
        _leg_record(scenario="release_via_ci", arm="proxy", note_variant="plain", seed=0, k=3, session_usd=0.12, metrics=_metrics(mistake_committed=False, turns_to_fact=2)),
    ]
    _write_results_jsonl(tmp_path, records)

    sc = report.scenario_report(tmp_path, "release_via_ci")
    assert set(sc["trajectories"]) == {"release_via_ci-none-none-s0", "release_via_ci-proxy-plain-s0"}
    assert sc["trajectories"]["release_via_ci-none-none-s0"]["mistake_repetition_rate"] == 1.0
    assert sc["trajectories"]["release_via_ci-proxy-plain-s0"]["mistake_rate_post"] == 0.0

    delta = sc["cross_arm_delta_vs_none"]["release_via_ci-proxy-plain-s0"]
    assert len(delta) == 2
    assert all(d["rdc_delta_a_minus_x"]["turns_to_fact"] == 4 for d in delta)
    assert all(d["session_usd_delta_a_minus_x"] == pytest.approx(0.18) for d in delta)


def test_scenario_report_filters_by_seed(tmp_path):
    _write_leg1(tmp_path, "release_via_ci", 0, mistake_committed=True)
    _write_leg1(tmp_path, "release_via_ci", 1, mistake_committed=True)
    records = [
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2),
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=1, k=2),
    ]
    _write_results_jsonl(tmp_path, records)
    sc = report.scenario_report(tmp_path, "release_via_ci", seed=1)
    assert set(sc["trajectories"]) == {"release_via_ci-none-none-s1"}


def test_build_report_covers_only_requested_scenarios(tmp_path):
    _write_results_jsonl(tmp_path, [])
    full = report.build_report(tmp_path)
    assert set(full) == set(scen.SCENARIOS)
    scoped = report.build_report(tmp_path, scenarios=["release_via_ci", "bench_box_only"])
    assert set(scoped) == {"release_via_ci", "bench_box_only"}


# ---------------------------------------------------------------------------
# DEFECT 9: contradictions surfaced in trajectory_report rows, and the
# `rescored=True` filename plumbing rescore.py's output relies on.
# ---------------------------------------------------------------------------


def test_trajectory_report_row_surfaces_contradictions_from_score():
    rec = _leg_record(scenario="deploy_reverted_by_reconciler", arm="none", note_variant="none", seed=0, k=2)
    rec["score"] = {
        "contradictions": [
            {
                "kind": "mistake_action_without_state_mutation",
                "mistake_state_check": "deploy_state_untouched",
                "detail": "action-stream and file-state evidence disagree",
            }
        ]
    }
    t = report.trajectory_report("traj", [rec], None, "told")
    row = next(r for r in t["legs"] if r["k"] == 2)
    assert row["contradictions"][0]["kind"] == "mistake_action_without_state_mutation"


def test_trajectory_report_row_contradictions_defaults_to_empty_list():
    """Every existing fixture in this file omits `score` entirely -- must not KeyError."""
    rec = _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2)
    assert "score" not in rec
    t = report.trajectory_report("traj", [rec], None, "discovered")
    row = next(r for r in t["legs"] if r["k"] == 2)
    assert row["contradictions"] == []


def test_format_human_readable_prints_contradiction_lines(tmp_path):
    rec = _leg_record(scenario="deploy_reverted_by_reconciler", arm="none", note_variant="none", seed=0, k=2)
    rec["score"] = {
        "contradictions": [
            {"kind": "mistake_action_without_state_mutation", "detail": "disagreement detail text"}
        ]
    }
    _write_results_jsonl(tmp_path, [rec])
    text = report.format_human_readable(report.build_report(tmp_path, scenarios=["deploy_reverted_by_reconciler"]))
    assert "CONTRADICTION" in text
    assert "disagreement detail text" in text


def test_build_report_rescored_reads_rescored_filenames_not_originals(tmp_path):
    """`rescore.py` writes `results.rescored.jsonl` / `result.rescored.json` as NEW
    sibling files, never touching the originals -- `build_report(rescored=True)`
    must read exclusively from those siblings.
    """
    _write_leg1(tmp_path, "release_via_ci", 0, mistake_committed=True)
    original = _leg_record(
        scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
        metrics=_metrics(mistake_committed=True),
    )
    _write_results_jsonl(tmp_path, [original])

    rescored_leg1_dir = tmp_path / "_shared" / "leg1" / "release_via_ci-s0"
    (rescored_leg1_dir / "result.rescored.json").write_text(
        json.dumps(_leg1_result(mistake_committed=False))
    )
    rescored_record = _leg_record(
        scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2,
        metrics=_metrics(mistake_committed=False),
    )
    (tmp_path / "results.rescored.jsonl").write_text(json.dumps(rescored_record) + "\n")

    before = report.build_report(tmp_path, scenarios=["release_via_ci"], rescored=False)
    after = report.build_report(tmp_path, scenarios=["release_via_ci"], rescored=True)

    before_traj = before["release_via_ci"]["trajectories"]["release_via_ci-none-none-s0"]
    after_traj = after["release_via_ci"]["trajectories"]["release_via_ci-none-none-s0"]
    before_k2 = next(r for r in before_traj["legs"] if r["k"] == 2)
    after_k2 = next(r for r in after_traj["legs"] if r["k"] == 2)
    assert before_k2["mistake_committed"] is True
    assert after_k2["mistake_committed"] is False
    assert before_traj["leg1_mistake_committed"] is True
    assert after_traj["leg1_mistake_committed"] is False


def test_format_human_readable_smoke(tmp_path):
    _write_leg1(tmp_path, "release_via_ci", 0, mistake_committed=True)
    _write_results_jsonl(tmp_path, [
        _leg_record(scenario="release_via_ci", arm="none", note_variant="none", seed=0, k=2, metrics=_metrics(mistake_committed=True)),
    ])
    text = report.format_human_readable(report.build_report(tmp_path, scenarios=["release_via_ci"]))
    assert "release_via_ci" in text
    assert "mistake_rate_post" in text


def test_load_results_jsonl_returns_empty_list_when_file_absent(tmp_path):
    assert report.load_results_jsonl(tmp_path) == []
