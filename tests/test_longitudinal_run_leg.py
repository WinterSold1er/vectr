"""Unit tests for `benchmarks/longitudinal_rediscovery/run_leg.py`'s own defects
fixed on branch `feature/eval-t1-validity-fixes`:

1. `LegRunner.plant_note()` re-snapshots `/v1/status` AFTER planting so
   `notes_count_at_start` (the scorer's non-vacuity premise counter, DESIGN.md 4.1)
   records what the agent will actually see -- not the stale pre-plant count taken
   in `start_daemon()`, before the plant step runs.
2. `main()` treats a session-level abort (`scorer.leg_non_vacuity`'s arm-agnostic
   `session_errored` gate) as a genuine abort: nonzero process exit, never chained
   or cached by a driver.

Neither test spawns a real daemon, proxy, or agent: `plant_note()`'s only external
calls are through the module-level `_http_json` helper, which is monkeypatched here;
the exit-on-session-error behavior is exercised directly against a constructed
`record` dict, the same shape `LegRunner.score()` produces.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "longitudinal_rediscovery"

# Same collision-safe loading pattern as test_longitudinal_plan.py / _scorer.py: two
# harnesses under benchmarks/ ship same-named scenarios.py/scorer.py files, so a bare
# `import` would race for one sys.modules slot. run_leg.py itself uses this identical
# `_vectr_eval_longitudinal_scenarios` key when it loads scenarios.py, so this test's
# `scen` and run_leg's own converge on ONE module object.
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
run_leg = _load_local_module("_vectr_eval_longitudinal_run_leg_test", "run_leg.py")
# Same key test_longitudinal_plan.py uses for run_plan.py -- not load-bearing here
# (these tests only call run_plan's pure path-resolution helpers), but keeping one
# converged module object per file avoids a second, independent import of a
# same-named module in the shared collision-safe registry.
run_plan = _load_local_module("_vectr_eval_longitudinal_run_plan_test", "run_plan.py")


def _make_runner(tmp_path: Path, **overrides) -> "run_leg.LegRunner":
    ap = run_leg._build_argparser()
    argv = [
        "--scenario", "bench_box_only",
        "--seed", "0",
        "--k", "2",
        "--leg-dir", str(tmp_path / "leg"),
        "--trajectory-id", "bench_box_only-proxy-plain-s0",
        "--arm", "proxy",
        "--note-variant", "plain",
        "--db-dir", str(tmp_path / "db"),
        "--daemon-port", "8899",
        "--proxy-port", "8900",
    ]
    args = ap.parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return run_leg.LegRunner(args)


# ---------------------------------------------------------------------------
# Defect 1: plant_note() re-snapshots notes_count_at_start AFTER planting
# ---------------------------------------------------------------------------


def test_plant_note_records_post_plant_note_count(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)
    # start_daemon() would normally set this from a pre-plant /v1/status read;
    # simulate that here without a real daemon.
    runner.notes_count_at_start = 0
    runner.record["notes_in_store_at_start"] = 0

    calls: list[tuple[str, str]] = []

    def fake_http_json(method, url, payload=None, timeout=30):
        calls.append((method, url))
        if method == "POST" and url.endswith("/v1/memory/clear"):
            # UPG-EVAL-STORE-MATCH: plant_note() resets the store first.
            return {"deleted": 0}
        if method == "POST" and url.endswith("/v1/remember"):
            return {"note_id": 7}
        if method == "GET" and url.endswith("/v1/status"):
            # Post-plant: the store now holds the just-planted note.
            return {"notes_count": 1}
        raise AssertionError(f"unexpected call: {method} {url}")

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)

    runner.plant_note()

    # The premise counter the scorer reads must reflect the POST-plant count...
    assert runner.notes_count_at_start == 1
    assert runner.record["notes_in_store_at_start"] == 1
    # ...and the stale pre-plant snapshot is preserved separately, not discarded.
    assert runner.record["notes_in_store_pre_plant"] == 0
    # One GET /v1/status re-snapshot happened, after the POST /v1/remember plant,
    # which itself happened after the store reset (UPG-EVAL-STORE-MATCH).
    assert ("POST", f"http://127.0.0.1:{runner.daemon_port}/v1/memory/clear") in calls
    assert ("POST", f"http://127.0.0.1:{runner.daemon_port}/v1/remember") in calls
    assert ("GET", f"http://127.0.0.1:{runner.daemon_port}/v1/status") in calls
    clear_i = calls.index(("POST", f"http://127.0.0.1:{runner.daemon_port}/v1/memory/clear"))
    plant_i = calls.index(("POST", f"http://127.0.0.1:{runner.daemon_port}/v1/remember"))
    status_i = calls.index(("GET", f"http://127.0.0.1:{runner.daemon_port}/v1/status"))
    assert clear_i < plant_i < status_i
    assert runner.record["note_store_reset_before_plant"] == 0


def test_plant_note_falls_back_to_pre_plant_count_on_bad_status_reply(tmp_path, monkeypatch):
    """A daemon hiccup on the diagnostic re-snapshot must not abort the leg over a
    read that isn't the plant itself -- notes_count_at_start keeps its last known
    (pre-plant) value rather than being corrupted by a non-int/error reply.
    """
    runner = _make_runner(tmp_path)
    runner.notes_count_at_start = 3
    runner.record["notes_in_store_at_start"] = 3

    def fake_http_json(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/memory/clear"):
            return {"deleted": 3}
        if method == "POST" and url.endswith("/v1/remember"):
            return {"note_id": 9}
        if method == "GET" and url.endswith("/v1/status"):
            return {"_error": "ConnectionResetError"}
        raise AssertionError(f"unexpected call: {method} {url}")

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)

    runner.plant_note()

    assert runner.notes_count_at_start == 3
    assert runner.record["notes_in_store_at_start"] == 3
    assert runner.record["notes_in_store_pre_plant"] == 3


# ---------------------------------------------------------------------------
# Defect 2: a session-level abort exits main() nonzero
# ---------------------------------------------------------------------------


def test_main_raises_systemexit_when_score_flags_session_errored():
    """`main()` runs `score()` (which calls `scorer.leg_non_vacuity`, setting
    `record["non_vacuity"]["session_errored"]`) then calls
    `_abort_if_session_errored(runner.record)` before returning. This test calls
    that real function directly against a constructed record shaped exactly like
    `score()` would leave it, so no daemon/proxy/agent is spawned.
    """
    errored_record = {
        "non_vacuity": {"session_errored": True},
        "invalid_reason": "agent session errored (is_error=True, agent_returncode=1, output_tokens=0)",
    }
    with pytest.raises(SystemExit) as excinfo:
        run_leg._abort_if_session_errored(errored_record)
    assert "agent session errored" in str(excinfo.value)

    clean_record = {
        "non_vacuity": {"session_errored": False},
        "invalid_reason": "",
    }
    # A clean session must never trip the same abort path (this is the negative
    # control -- the whole point of scoping the abort to session_errored alone,
    # not to every invalid leg: a task-failing-but-ran leg is still a completed
    # leg for resumability purposes, exit 0).
    run_leg._abort_if_session_errored(clean_record)  # must not raise

    # Also: a record with no non_vacuity key at all (e.g. a code path that never
    # reached score()) must not spuriously abort.
    run_leg._abort_if_session_errored({})  # must not raise


def test_leg_non_vacuity_session_errored_produces_the_reason_main_reads():
    """Ties the scorer's `session_errored` gate to the exact record shape
    `run_leg.py`'s `main()` inspects (`record["non_vacuity"]["session_errored"]`,
    `record["invalid_reason"]`) -- the true end-to-end contract, without spawning a
    live agent. See test_longitudinal_scorer.py for the scorer-level discrimination
    tests of this same gate.
    """
    scorer = _load_local_module("_vectr_eval_longitudinal_scorer_test", "scorer.py")
    events = [
        {"type": "system", "subtype": "init", "mcp_servers": [], "tools": []},
        {"type": "result", "subtype": "success", "num_turns": 1, "usage": {}, "is_error": True},
    ]
    nv = scorer.leg_non_vacuity(
        arm="none", k=1, events=events, notes_count_at_start=0,
        agent_returncode=1, is_error=True, output_tokens=0,
    )
    record = {"non_vacuity": nv["non_vacuity"], "invalid_reason": nv["invalid_reason"], "valid": nv["valid"]}
    assert record["valid"] is False
    assert record["non_vacuity"]["session_errored"] is True
    should_abort = (record.get("non_vacuity") or {}).get("session_errored")
    assert should_abort


# ---------------------------------------------------------------------------
# DEFECT 4 (branch feature/eval-workspace-stable): --workspace-dir makes the
# agent/daemon workspace trajectory-stable instead of nested under the per-leg
# --leg-dir, so a note leg k plants is registered under the same workspace path
# leg k+1's fresh daemon queries (vectr's working-memory store scopes notes by
# workspace path, not by --db-dir alone -- see run_leg.py's module docstring and
# `--workspace-dir` help text).
# ---------------------------------------------------------------------------


def test_workspace_defaults_to_leg_dir_nested_when_workspace_dir_omitted(tmp_path):
    """Unchanged pre-fix behavior for any caller that never passes
    --workspace-dir (standalone k==1 debugging, T0's single-shot probe cells):
    the workspace stays nested under this leg's own --leg-dir, exactly as before.
    """
    runner = _make_runner(tmp_path)
    assert runner.workspace == runner.root / "workspace"
    assert runner.workspace == (tmp_path / "leg" / "workspace")


def test_workspace_dir_flag_overrides_to_a_trajectory_stable_path(tmp_path):
    """The actual fix: an explicit --workspace-dir (what run_plan.py now always
    passes for a normal leg) is used verbatim instead of the per-leg default, and
    is independent of --leg-dir/--k entirely.
    """
    traj_workspace = tmp_path / "traj" / "workspace"
    runner_k2 = _make_runner(
        tmp_path, k=2, leg_dir=str(tmp_path / "traj" / "legs" / "2"),
        workspace_dir=str(traj_workspace),
    )
    runner_k3 = _make_runner(
        tmp_path, k=3, leg_dir=str(tmp_path / "traj" / "legs" / "3"),
        workspace_dir=str(traj_workspace),
    )
    # Same workspace path for k=2 and k=3 of one trajectory (the invariant this
    # whole fix exists to establish)...
    assert runner_k2.workspace == runner_k3.workspace == traj_workspace
    # ...while each leg's own artifacts stay per-leg, unaffected.
    assert runner_k2.root != runner_k3.root
    assert runner_k2.artifacts == tmp_path / "traj" / "legs" / "2" / "artifacts"
    assert runner_k3.artifacts == tmp_path / "traj" / "legs" / "3" / "artifacts"


def test_workspace_dir_rejected_with_shared_leg1():
    ap = run_leg._build_argparser()
    args = ap.parse_args([
        "--shared-leg1", "--scenario", "bench_box_only", "--out-dir", "/tmp/x",
        "--workspace-dir", "/tmp/y",
    ])
    with pytest.raises(SystemExit):
        run_leg._validate_args(ap, args)


def test_reset_workspace_wipes_stale_content_left_by_a_prior_leg(tmp_path):
    """The other half of the fix: reusing one workspace path across legs is only
    safe because `prepare()`'s `_reset_workspace()` wipes it first -- otherwise a
    file an earlier leg's agent created (and that is absent from this leg's own
    restore-tar / fresh materialize) would wrongly persist into a later leg.
    """
    runner = _make_runner(tmp_path, workspace_dir=str(tmp_path / "traj-workspace"))
    runner.workspace.mkdir(parents=True)
    stray = runner.workspace / "leftover_from_prior_leg.txt"
    stray.write_text("should not survive")
    assert stray.exists()

    runner._reset_workspace()

    assert runner.workspace.is_dir()
    assert not stray.exists()
    assert list(runner.workspace.iterdir()) == []


def test_reset_workspace_is_a_no_op_the_first_time(tmp_path):
    """A never-before-used workspace path (first leg of a trajectory) has nothing
    to wipe -- must not raise on a directory that does not exist yet.
    """
    runner = _make_runner(tmp_path, k=1, workspace_dir=str(tmp_path / "brand-new"))
    assert not runner.workspace.exists()
    runner._reset_workspace()
    assert runner.workspace.is_dir()
    assert list(runner.workspace.iterdir()) == []


# ---------------------------------------------------------------------------
# DEFECT 6 (branch feature/eval-hook-preflight): zero-cost SessionStart hook
# mechanism preflight, run before the paid `claude -p` session for arms
# hook-sessionstart/hook-full. None of these tests spawn a real daemon or hook
# subprocess: `_http_json` (daemon calls) and `_run_hook_command` (the hook
# subprocess spawn) are the two module-level seams, both monkeypatched here --
# the exact same pattern the existing plant_note() tests above use for `_http_json`.
# ---------------------------------------------------------------------------


def _make_hook_runner(tmp_path: Path, *, arm: str = "hook-sessionstart", **overrides) -> "run_leg.LegRunner":
    overrides.setdefault("planted_note_id", 7)
    overrides.setdefault("planted_anchor", "note:7")
    runner = _make_runner(tmp_path, arm=arm, **overrides)
    runner.workspace.mkdir(parents=True, exist_ok=True)
    runner.artifacts.mkdir(parents=True, exist_ok=True)
    settings_path = runner.workspace / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({
        "hooks": {
            "SessionStart": [{
                "matcher": "startup|resume|clear|compact",
                "hooks": [{"type": "command", "command": "fake-vectr-hook session-start"}],
            }],
        },
    }))
    runner.record["hooks_settings_path"] = str(settings_path)
    return runner


def _note_content(runner: "run_leg.LegRunner") -> str:
    return runner._find_note_variant().content


def test_hook_preflight_noop_for_non_hook_arms(tmp_path, monkeypatch):
    """arm "proxy" (or "none"/"mcp"/"mcp-bare") must be completely unaffected --
    no daemon call, no hook subprocess spawn, no record mutation.
    """
    runner = _make_runner(tmp_path)  # default arm="proxy" per _make_runner
    assert runner.arm == "proxy"

    def fail_http(*a, **k):
        raise AssertionError("hook_preflight must not touch the daemon for a non-hook arm")

    def fail_run(*a, **k):
        raise AssertionError("hook_preflight must not spawn a hook subprocess for a non-hook arm")

    monkeypatch.setattr(run_leg, "_http_json", fail_http)
    monkeypatch.setattr(run_leg, "_run_hook_command", fail_run)

    runner.hook_preflight()  # must return immediately, no exception

    assert "hook_preflight" not in runner.record


def test_hook_preflight_noop_for_hook_userpromptsubmit_arm(tmp_path, monkeypatch):
    """arm "hook-userpromptsubmit" IS a hook arm, but `hook_preflight()` stays
    scoped exactly to ("hook-sessionstart", "hook-full") -- it is a
    SessionStart-specific mechanism check (synthesizes a SessionStart hook
    stdin payload and greps its stdout). The new arm's own firing/delivery
    evidence comes from `run_agent()`'s before/after
    `hook_injection_counts['UserPromptSubmit']` delta instead (DESIGN.md 4.1),
    never from this method."""
    runner = _make_hook_runner(tmp_path, arm="hook-userpromptsubmit")

    def fail_http(*a, **k):
        raise AssertionError("hook_preflight must not touch the daemon for arm hook-userpromptsubmit")

    def fail_run(*a, **k):
        raise AssertionError("hook_preflight must not spawn a hook subprocess for arm hook-userpromptsubmit")

    monkeypatch.setattr(run_leg, "_http_json", fail_http)
    monkeypatch.setattr(run_leg, "_run_hook_command", fail_run)

    runner.hook_preflight()  # must return immediately, no exception

    assert "hook_preflight" not in runner.record


def test_hook_preflight_noop_when_nothing_planted(tmp_path, monkeypatch):
    """A hook arm with no planted note yet (e.g. leg 1) has nothing to preflight."""
    runner = _make_hook_runner(tmp_path, planted_note_id=None, planted_anchor=None)
    assert runner.note_id is None and runner.planted_anchor is None

    def fail_http(*a, **k):
        raise AssertionError("must not touch the daemon with nothing planted")

    monkeypatch.setattr(run_leg, "_http_json", fail_http)
    runner.hook_preflight()
    assert "hook_preflight" not in runner.record


def test_hook_preflight_success_records_evidence_and_does_not_abort(tmp_path, monkeypatch):
    runner = _make_hook_runner(tmp_path)
    content = _note_content(runner)

    status_calls = {"n": 0}

    def fake_http_json(method, url, payload=None, timeout=30):
        assert url.startswith(f"http://127.0.0.1:{runner.daemon_port}/")
        assert method == "GET" and url.endswith("/v1/status")
        status_calls["n"] += 1
        # Before the hook call: 0 injections; after: 1 -- a real delta.
        count = 0 if status_calls["n"] == 1 else 1
        return {"hook_injection_counts": {"SessionStart": count}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        assert command == "fake-vectr-hook session-start"
        assert cwd == str(runner.workspace)
        payload = json.loads(stdin)
        assert payload["cwd"] == str(runner.workspace)
        assert payload["hook_event_name"] == "SessionStart"
        stdout = json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": content}
        })
        return 0, stdout, "", False

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)

    runner.hook_preflight()  # must not raise

    hp = runner.record["hook_preflight"]
    assert hp["daemon_evidence"] is True
    assert hp["session_start_delta"] == 1
    assert hp["stdout_has_planted_content"] is True
    assert hp["command"] == "fake-vectr-hook session-start"
    assert (runner.artifacts / "hook-preflight.json").is_file()


def test_hook_preflight_aborts_when_daemon_evidence_missing(tmp_path, monkeypatch):
    """The hook subprocess reports success and echoes the right content, but this
    leg's own scratch daemon never sees hook_injection_counts move -- the exact
    live symptom this preflight exists to catch pre-spend (DEFECT 6).
    """
    runner = _make_hook_runner(tmp_path)
    content = _note_content(runner)

    def fake_http_json(method, url, payload=None, timeout=30):
        # Same (zero) count before and after -- no daemon-side evidence.
        return {"hook_injection_counts": {}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        stdout = json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": content}
        })
        return 0, stdout, "", False

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)

    with pytest.raises(SystemExit) as excinfo:
        runner.hook_preflight()
    msg = str(excinfo.value)
    assert "daemon_evidence=False" in msg
    assert "fix the scenario" in msg
    # The artifact is still written even on abort, for post-mortem.
    hp = runner.record["hook_preflight"]
    assert hp["daemon_evidence"] is False
    assert hp["stdout_has_planted_content"] is True


def test_hook_preflight_aborts_when_stdout_content_missing(tmp_path, monkeypatch):
    """Daemon-side evidence is real (the count moves), but the hook's own stdout
    never carried the planted note's content -- e.g. a kind/trigger-eligibility
    gap where the delivery mechanism fires but returns empty/unrelated text.
    """
    runner = _make_hook_runner(tmp_path)

    status_calls = {"n": 0}

    def fake_http_json(method, url, payload=None, timeout=30):
        status_calls["n"] += 1
        count = 0 if status_calls["n"] == 1 else 1
        return {"hook_injection_counts": {"SessionStart": count}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        return 0, "", "", False  # empty stdout -- e.g. a silently-ineligible note

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)

    with pytest.raises(SystemExit) as excinfo:
        runner.hook_preflight()
    msg = str(excinfo.value)
    assert "stdout_has_planted_content=False" in msg
    hp = runner.record["hook_preflight"]
    assert hp["daemon_evidence"] is True
    assert hp["stdout_has_planted_content"] is False


def test_hook_preflight_wrong_daemon_resolution_abort_names_the_culprit_port(tmp_path, monkeypatch):
    """Instance mis-resolution (the hook command resolves, via the global
    ~/.vectr/instances.json registry, to a DIFFERENT daemon than this leg's own
    scratch port) is caught by the SAME daemon_evidence assertion (the count never
    moves on THIS leg's own daemon) -- this test additionally checks the registry
    diagnostic names the wrong port in the abort message instead of leaving it a
    mystery.
    """
    runner = _make_hook_runner(tmp_path)
    content = _note_content(runner)
    other_port = runner.daemon_port + 1

    registry_path = tmp_path / "instances.json"
    registry_path.write_text(json.dumps({
        run_leg._workspace_hash(str(runner.workspace.resolve())): {"port": other_port},
    }))
    monkeypatch.setattr(run_leg, "_VECTR_INSTANCE_REGISTRY_PATH", registry_path)

    def fake_http_json(method, url, payload=None, timeout=30):
        # This leg's own daemon never sees the increment (the hook actually
        # talked to `other_port`, simulated here simply as "no evidence").
        return {"hook_injection_counts": {}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        stdout = json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": content}
        })
        return 0, stdout, "", False

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)

    with pytest.raises(SystemExit) as excinfo:
        runner.hook_preflight()
    msg = str(excinfo.value)
    assert f"port {other_port}" in msg
    assert "mis-resolution" in msg
    hp = runner.record["hook_preflight"]
    assert hp["registry_resolved_port"] == other_port
    assert hp["registry_port_matches_leg_daemon"] is False


def test_hook_preflight_allow_hook_unreachable_bypasses_abort(tmp_path, monkeypatch):
    runner = _make_hook_runner(tmp_path, allow_hook_unreachable=True)

    def fake_http_json(method, url, payload=None, timeout=30):
        return {"hook_injection_counts": {}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        return 0, "", "", False

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)

    runner.hook_preflight()  # must not raise despite both assertions failing

    hp = runner.record["hook_preflight"]
    assert hp["daemon_evidence"] is False
    assert hp["stdout_has_planted_content"] is False


def test_run_agent_passes_settings_for_hook_arms_only(tmp_path, monkeypatch):
    """`--settings <hooks_settings_path>` is added to the spawned `claude` argv
    for arms hook-sessionstart/hook-full/hook-userpromptsubmit (so headless
    hook loading no longer depends on directory trust), and left out for
    every other arm.
    """
    captured = {}

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    # Both hook arms exercised below bracket the spawned agent with two
    # `/v1/status` GETs against the leg's daemon port. Only argv is under
    # test here, but without this stub those are LIVE connections to
    # 127.0.0.1:8899 — whether anything answers is a property of the
    # developer's machine, not of the test. Same stub the delta tests below
    # use; caught by the conftest socket guard.
    monkeypatch.setattr(
        run_leg,
        "_http_json",
        lambda method, url, payload=None, timeout=30: {
            "hook_injection_counts": {"SessionStart": 0, "UserPromptSubmit": 0}
        },
    )
    monkeypatch.setattr(run_leg.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_leg.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    hook_runner = _make_hook_runner(tmp_path)
    hook_runner.run_agent()
    assert "--settings" in captured["cmd"]
    idx = captured["cmd"].index("--settings")
    assert captured["cmd"][idx + 1] == hook_runner.record["hooks_settings_path"]

    usps_runner = _make_hook_runner(tmp_path, arm="hook-userpromptsubmit")
    usps_runner.run_agent()
    assert "--settings" in captured["cmd"]
    idx = captured["cmd"].index("--settings")
    assert captured["cmd"][idx + 1] == usps_runner.record["hooks_settings_path"]

    proxy_runner = _make_runner(tmp_path)  # arm="proxy"
    proxy_runner.workspace.mkdir(parents=True, exist_ok=True)
    proxy_runner.artifacts.mkdir(parents=True, exist_ok=True)
    proxy_runner.run_agent()
    assert "--settings" not in captured["cmd"]


def test_run_agent_captures_user_prompt_submit_injection_delta_for_usps_arm(tmp_path, monkeypatch):
    """DESIGN.md 4.1 / task item 3: arm "hook-userpromptsubmit" has no
    SessionStart or PreToolUse hook to preflight, and its additionalContext
    never renders into the stream-json transcript, so `run_agent()` itself
    must bracket the spawned agent subprocess with two `/v1/status` GETs and
    record the `hook_injection_counts['UserPromptSubmit']` delta -- the sole
    firing/delivery evidence `scorer.leg_non_vacuity` uses for this arm.
    """
    status_responses = iter([
        {"hook_injection_counts": {"UserPromptSubmit": 3}},   # before
        {"hook_injection_counts": {"UserPromptSubmit": 5}},   # after
    ])

    def fake_http_json(method, url, payload=None, timeout=30):
        assert method == "GET" and url.endswith("/v1/status")
        return next(status_responses)

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg.subprocess, "Popen", lambda cmd, **kwargs: _FakeProc())
    monkeypatch.setattr(run_leg.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    runner = _make_hook_runner(tmp_path, arm="hook-userpromptsubmit")
    runner.run_agent()

    assert runner.record["user_prompt_submit_injection_count_before"] == 3
    assert runner.record["user_prompt_submit_injection_count_after"] == 5
    assert runner.record["user_prompt_submit_injection_delta"] == 2


def test_run_agent_flat_counter_yields_zero_delta_for_usps_arm(tmp_path, monkeypatch):
    """The counter never moving (the hook did not fire this leg) must record
    a delta of 0, not raise or silently omit the field -- `scorer.
    leg_non_vacuity` treats < 1 as invalid."""

    def fake_http_json(method, url, payload=None, timeout=30):
        return {"hook_injection_counts": {"UserPromptSubmit": 4}}

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg.subprocess, "Popen", lambda cmd, **kwargs: _FakeProc())
    monkeypatch.setattr(run_leg.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    runner = _make_hook_runner(tmp_path, arm="hook-userpromptsubmit")
    runner.run_agent()

    assert runner.record["user_prompt_submit_injection_delta"] == 0


def test_run_agent_does_not_capture_usps_delta_for_other_arms(tmp_path, monkeypatch):
    """arm "proxy" (and every other non-hook arm) is completely untouched:
    `run_agent()` must not call `/v1/status` at all for it, and must never
    write any hook-delta key (USPS or SessionStart) into `self.record`.

    UPG-EVAL-HOOKCOUNT-CUMULATIVE revision: arms "hook-sessionstart"/
    "hook-full" are no longer "other arms" for this specific purpose --
    `run_agent()` now DOES bracket its own subprocess with before/after
    `/v1/status` snapshots for these two arms too (see the dedicated
    `hook_injection_counts` delta tests above), so this test only asserts
    they never pick up the USPS-SPECIFIC keys, which stay scoped to arm
    "hook-userpromptsubmit" alone.
    """
    status_responses = iter([
        {"hook_injection_counts": {"SessionStart": 1}},  # hook-sessionstart: before
        {"hook_injection_counts": {"SessionStart": 1}},  # hook-sessionstart: after (flat -> delta 0)
        {"hook_injection_counts": {"SessionStart": 1}},  # hook-full: before
        {"hook_injection_counts": {"SessionStart": 1}},  # hook-full: after (flat -> delta 0)
    ])

    def fake_http_json(method, url, payload=None, timeout=30):
        assert method == "GET" and url.endswith("/v1/status")
        return next(status_responses)

    def fail_http(*a, **k):
        raise AssertionError("run_agent() must not call /v1/status for arm 'proxy'")

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(run_leg.subprocess, "Popen", lambda cmd, **kwargs: _FakeProc())
    monkeypatch.setattr(run_leg.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    for arm in ("hook-sessionstart", "hook-full"):
        runner = _make_hook_runner(tmp_path, arm=arm)
        runner.run_agent()
        assert "user_prompt_submit_injection_delta" not in runner.record
        assert runner.record["hook_injection_counts"] == {"SessionStart": 0}

    monkeypatch.setattr(run_leg, "_http_json", fail_http)
    proxy_runner = _make_runner(tmp_path)  # arm="proxy"
    proxy_runner.workspace.mkdir(parents=True, exist_ok=True)
    proxy_runner.artifacts.mkdir(parents=True, exist_ok=True)
    proxy_runner.run_agent()
    assert "user_prompt_submit_injection_delta" not in proxy_runner.record
    assert "hook_injection_counts" not in proxy_runner.record


# ---------------------------------------------------------------------------
# write_hooks(): `vectr init --hooks` (main.py's `_write_claude_hooks`) always
# writes all six hook groups; write_hooks() then prunes `.claude/settings.json`
# down to the single group the arm actually needs. Fixture below mirrors the
# real six-group shape main.py produces (matchers/commands taken from
# main.py::_write_claude_hooks) so the prune logic is exercised against a
# realistic pre-prune file, not a synthetic one-group stand-in.
# ---------------------------------------------------------------------------

_FULL_SIX_GROUP_HOOKS = {
    "SessionStart": [{
        "matcher": "startup|resume|clear|compact",
        "hooks": [{"type": "command", "command": "vectr hook session-start"}],
    }],
    "UserPromptSubmit": [{
        "hooks": [{"type": "command", "command": "vectr hook user-prompt-submit"}],
    }],
    "PreToolUse": [{
        "matcher": "Edit|Write|Read|Bash",
        "hooks": [{"type": "command", "command": "vectr hook pre-tool-use"}],
    }],
    "PostToolUse": [{
        "matcher": "Bash|Edit|Write|MultiEdit",
        "hooks": [{"type": "command", "command": "vectr hook post-tool-use"}],
    }],
    "PostToolUseFailure": [{
        "matcher": "Bash|Edit|Write|MultiEdit",
        "hooks": [{"type": "command", "command": "vectr hook post-tool-use"}],
    }],
    "PreCompact": [{
        "matcher": "manual|auto",
        "hooks": [{"type": "command", "command": "vectr hook pre-compact"}],
    }],
}


def _fake_vectr_init_hooks_run(monkeypatch, tmp_path):
    """Monkeypatches `run_leg.subprocess.run` (the `vectr init --hooks` spawn
    inside `write_hooks()`) so it writes a realistic full six-group
    `.claude/settings.json`, exactly as `vectr init --hooks --no-ide-config`
    would, without actually invoking the CLI."""

    def fake_run(cmd, **kwargs):
        assert "--hooks" in cmd
        path_idx = cmd.index("--path")
        workspace = Path(cmd[path_idx + 1])
        settings_dir = workspace / ".claude"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({"hooks": _FULL_SIX_GROUP_HOOKS}, indent=2)
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="wrote hooks\n", stderr="")

    monkeypatch.setattr(run_leg.subprocess, "run", fake_run)


def test_write_hooks_prunes_to_user_prompt_submit_only_for_usps_arm(tmp_path, monkeypatch):
    """Task item 1/6: arm "hook-userpromptsubmit" gets a settings.json whose
    hooks dict has ONLY the "UserPromptSubmit" key -- no SessionStart hook, no
    PreToolUse hook, wired at all."""
    _fake_vectr_init_hooks_run(monkeypatch, tmp_path)
    runner = _make_runner(tmp_path, arm="hook-userpromptsubmit")
    runner.workspace.mkdir(parents=True, exist_ok=True)
    runner.artifacts.mkdir(parents=True, exist_ok=True)

    runner.write_hooks()

    settings_path = runner.workspace / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    assert list(data["hooks"].keys()) == ["UserPromptSubmit"]
    assert runner.record["hooks_pruned_to"] == ["UserPromptSubmit"]
    assert runner.record["hooks_settings_path"] == str(settings_path)


def test_write_hooks_prunes_to_session_start_only_for_hook_sessionstart_arm(tmp_path, monkeypatch):
    """Companion coverage for the pre-existing D1 pruning path -- there was no
    test for write_hooks() at all before the hook-userpromptsubmit arm was
    added (a pre-existing gap in this file), so this closes it for D1 too."""
    _fake_vectr_init_hooks_run(monkeypatch, tmp_path)
    runner = _make_runner(tmp_path, arm="hook-sessionstart")
    runner.workspace.mkdir(parents=True, exist_ok=True)
    runner.artifacts.mkdir(parents=True, exist_ok=True)

    runner.write_hooks()

    settings_path = runner.workspace / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    assert list(data["hooks"].keys()) == ["SessionStart"]
    assert runner.record["hooks_pruned_to"] == ["SessionStart"]


def test_write_hooks_does_not_prune_for_hook_full_arm(tmp_path, monkeypatch):
    """arm "hook-full" needs the whole set (SessionStart + PreToolUse +
    UserPromptSubmit, attestation-gated) -- write_hooks() must leave every
    group `vectr init --hooks` produced untouched and must not set
    `hooks_pruned_to` at all."""
    _fake_vectr_init_hooks_run(monkeypatch, tmp_path)
    runner = _make_runner(tmp_path, arm="hook-full")
    runner.workspace.mkdir(parents=True, exist_ok=True)
    runner.artifacts.mkdir(parents=True, exist_ok=True)

    runner.write_hooks()

    settings_path = runner.workspace / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    assert set(data["hooks"].keys()) == set(_FULL_SIX_GROUP_HOOKS.keys())
    assert "hooks_pruned_to" not in runner.record


# ---------------------------------------------------------------------------
# DEFECT 6 follow-up: plant-time hook-channel delivery metadata
# (`LegRunner._apply_hook_delivery_metadata()`). `plant_note()` always sends
# an EXPLICIT `triggers` list (every `NoteVariant.trigger_paths` is
# non-empty), which per `trigger_engine.effective_triggers()`'s
# replace-not-merge contract means the note's `kind` default trigger bundle
# never applies -- and the explicit list is PATH-only, which
# `trigger_engine._trigger_matches()` can never match at SessionStart/boot
# (no `file_path` is ever supplied there, so a path-bearing trigger
# deterministically fails). `_apply_hook_delivery_metadata()` closes this for
# arms "hook-sessionstart"/"hook-full" only by (1) appending an explicit
# `{"event": "session-start"}` trigger and (2) switching `kind` to
# "directive" (gotcha's 100-token per-kind injection cap is smaller than
# every scenario's real full-block render — 101-181 tokens measured directly
# against the shipped renderer — so a gotcha-kind hook note would always
# degrade to its index-tier, title-only line; directive's 400-token cap is
# not). Both empirically verified live against a scratch daemon (real
# `vectr hook session-start` + `vectr hook pre-tool-use`) before being
# encoded here — see the coder-lane report for this follow-up.
# ---------------------------------------------------------------------------


def _trigger_is_session_start_eligible(trigger: dict) -> bool:
    """Conservative, LOCAL mirror of `agent/trigger_engine.py`'s real
    `_trigger_matches()` conjunction check, specialized to the SessionStart/
    boot lifecycle moment — the same pattern
    `test_longitudinal_scorer.py::_probe_file_structurally_reaches` uses to
    mirror `recall_for_path()` rather than importing the product module (so
    this test stays a $0, offline, no-daemon structural check).

    At boot (`app/service.py::_recall_impl`'s `boot` branch ->
    `WorkingContextStore.fire_and_format(events=["session-start"], ...)`), no
    file_path/query/command/resolved-symbols context is ever supplied
    (nothing is being edited, run, or searched yet) — so the real
    `_trigger_matches()` structurally rejects any trigger declaring 'path',
    'symbol', 'semantic', or 'command' (each of those axes requires
    call-time context absent at boot), and accepts only a trigger whose
    'event' is unset or literally "session-start". `validate_trigger()`
    requires at least one axis, so a trigger meeting this rule always has
    `event == "session-start"` with no other axis set.
    """
    return (
        trigger.get("path") is None
        and trigger.get("symbol") is None
        and not trigger.get("semantic")
        and trigger.get("command") is None
        and trigger.get("event") == "session-start"
    )


def _hook_plant_payload(variant: "scen.NoteVariant") -> dict:
    """The subset of `plant_note()`'s payload construction that
    `_apply_hook_delivery_metadata()` reads/mutates (`triggers`, `kind`) —
    built the identical way `plant_note()` itself builds it, so this test
    exercises the real transformation, not a re-description of it."""
    payload: dict = {"kind": variant.kind}
    if variant.trigger_paths:
        payload["triggers"] = [{"path": p} for p in variant.trigger_paths]
    return payload


def test_every_note_variant_is_session_start_eligible_for_hook_arms(tmp_path):
    """Regression guard for DEFECT 6 (a live hook-sessionstart leg came back
    INVALID with `hook_injection_counts: {}`, no planted content delivered):
    the hook-arm analogue of
    `test_longitudinal_scorer.py::test_every_leg_probe_file_structurally_reaches_every_note_variant`.
    For every scenario, every note variant, both hook arms: the `triggers`
    list `_apply_hook_delivery_metadata()` produces must contain at least one
    session-start-eligible trigger (per the real trigger engine's own
    matching rule, mirrored above), and `kind` must be a kind whose per-kind
    injection budget the real content actually fits — asserted here as
    "directive", not "gotcha", for the reason documented on
    `_apply_hook_delivery_metadata()` itself. No daemon, no network call:
    pure structural evaluation over `scenarios.py`'s real data — the $0 test
    that would have caught DEFECT 6 before any `claude -p` spend.
    """
    for arm in ("hook-sessionstart", "hook-full"):
        runner = _make_runner(tmp_path, arm=arm)
        for slug, scenario in scen.SCENARIOS.items():
            for variant in scenario.note_variants:
                payload = _hook_plant_payload(variant)
                runner._apply_hook_delivery_metadata(payload)
                triggers = payload.get("triggers") or []
                assert any(_trigger_is_session_start_eligible(t) for t in triggers), (
                    f"arm={arm} {slug}/{variant.variant}: no session-start-eligible "
                    f"trigger in {triggers!r} — the SessionStart hook channel would "
                    f"silently deliver nothing for this note (DEFECT 6)"
                )
                assert payload["kind"] == "directive", (
                    f"arm={arm} {slug}/{variant.variant}: hook-arm kind is "
                    f"{payload['kind']!r}, not 'directive' — gotcha's 100-token "
                    f"per-kind injection cap is smaller than every scenario's "
                    f"real full-block render (101-181 tokens measured), so "
                    f"content would silently degrade to its index-tier, "
                    f"title-only line regardless of trigger eligibility"
                )


def test_apply_hook_delivery_metadata_is_a_noop_for_non_hook_arms(tmp_path):
    """Channel parity (DESIGN.md 8): every arm OTHER than hook-sessionstart/
    hook-full must be byte-for-byte unaffected by
    `_apply_hook_delivery_metadata()` — the advisory text and its delivery
    metadata are identical to today's shape for `none`/`mcp`/`mcp-bare`/
    `proxy`. `hook-userpromptsubmit` is included here too even though it IS a
    hook arm: it delivers via `_recall_impl`'s ordinary ranked `recall()`
    pass, never trigger-gated or kind-budget-capped the way SessionStart's
    boot branch is, so it needs neither the appended trigger nor the
    widened content budget (DESIGN.md 8's own hook-channel delivery
    metadata paragraph)."""
    for arm in ("none", "mcp", "mcp-bare", "proxy", "hook-userpromptsubmit"):
        runner = _make_runner(tmp_path, arm=arm)
        payload = {"kind": "gotcha", "triggers": [{"path": "**/foo.py"}]}
        before = json.loads(json.dumps(payload))
        runner._apply_hook_delivery_metadata(payload)
        assert payload == before


def test_plant_note_sends_directive_kind_and_session_start_trigger_for_hook_arms(tmp_path, monkeypatch):
    """End-to-end through `plant_note()` itself (not just the metadata helper
    in isolation): the real POST /v1/remember payload for a hook arm carries
    `kind="directive"` and a `{"event": "session-start"}` trigger alongside
    the scenario's own path triggers; a non-hook arm's payload is unchanged
    from before this follow-up (still `kind="gotcha"`, path-only triggers).
    """
    captured = {}

    def fake_http_json(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/memory/clear"):
            return {"deleted": 0}
        if method == "POST" and url.endswith("/v1/remember"):
            captured["payload"] = payload
            return {"note_id": 7}
        if method == "GET" and url.endswith("/v1/status"):
            return {"notes_count": 1}
        raise AssertionError(f"unexpected call: {method} {url}")

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)

    hook_runner = _make_runner(tmp_path, arm="hook-sessionstart")
    hook_runner.notes_count_at_start = 0
    hook_runner.record["notes_in_store_at_start"] = 0
    hook_runner.plant_note()
    hook_payload = captured["payload"]
    assert hook_payload["kind"] == "directive"
    assert {"event": "session-start"} in hook_payload["triggers"]
    assert any("path" in t for t in hook_payload["triggers"]), (
        "the pre-existing path-anchored triggers must survive alongside the "
        "added session-start trigger (OR composition), not be replaced"
    )

    proxy_runner = _make_runner(tmp_path, arm="proxy")
    proxy_runner.notes_count_at_start = 0
    proxy_runner.record["notes_in_store_at_start"] = 0
    proxy_runner.plant_note()
    proxy_payload = captured["payload"]
    assert proxy_payload["kind"] == "gotcha"
    assert {"event": "session-start"} not in proxy_payload["triggers"]


# ---------------------------------------------------------------------------
# DEFECT 7: `probe()`'s reachability ABORT criterion must be arm-aware. A live
# T2 re-run aborted at $0 -- both hook-sessionstart k2 legs failed
# `probe()` (not `hook_preflight()`, which runs later and already proves
# session-start delivery works) with "the planted note is not retrievable on
# EITHER daemon-side probe". Root cause: `_apply_hook_delivery_metadata()`
# (DEFECT 6 follow-up) plants hook-arm notes with kind="directive" to fit the
# content-injection budget, but agent/config.yaml's `proactive.
# structural_kinds` omits "directive" and `proactive.proxy.
# exclude_directive_notes` independently excludes it too -- so
# `_proactive_probe`'s channel (`/v1/proactive`, channel="proxy") can never
# see a directive-kind note, regardless of whether the SessionStart channel
# the hook arm actually tests can. `probe()` now judges hook-arm reachability
# against `_session_start_probe()` (a direct `boot=True, hook_event=
# "SessionStart"` `/v1/recall` call -- the same payload shape
# `agent/hook_cli.py::run_hook`'s "session-start" branch sends) instead, while
# still recording `_proactive_probe`'s result as a diagnostic-only value.
# Non-hook arms are unaffected: `_proactive_probe` remains their sole
# reachability channel, unchanged.
# ---------------------------------------------------------------------------


def _fake_http_json_for_probe(*, proactive_anchor_present: bool, session_start_notes: str | None):
    """Builds a `_http_json` stand-in that dispatches on URL suffix, returning
    REAL-shaped `/v1/proactive` and `/v1/recall` (boot) responses -- never the
    `[]`/`{}` placeholder pattern the coder brief's mock-fidelity rule forbids.
    `session_start_notes=None` means the daemon's boot recall returned nothing
    (an empty string, exactly `_recall_impl`'s own boot branch on a fresh/
    ineligible workspace)."""

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            anchor_ids = ["note:7"] if proactive_anchor_present else []
            return {"item_count": len(anchor_ids), "context": "", "anchor_ids": anchor_ids}
        if method == "POST" and url.endswith("/v1/recall"):
            assert payload.get("boot") is True and payload.get("hook_event") == "SessionStart", (
                f"probe()'s session-start channel call must send the exact "
                f"agent/hook_cli.py::run_hook 'session-start' branch payload shape, "
                f"got {payload!r}"
            )
            return {"notes": session_start_notes or ""}
        raise AssertionError(f"unexpected call in probe(): {method} {url}")

    return fake


def test_probe_hook_arm_judges_reachability_via_session_start_channel_not_proactive(tmp_path, monkeypatch):
    """Reproduces DEFECT 7's fix: the proactive channel is empty (exactly what
    a directive-kind note structurally produces) but the session-start channel
    carries the planted content -- `probe()` must NOT abort, and must record
    which channel it actually judged."""
    runner = _make_hook_runner(tmp_path, arm="hook-sessionstart")
    content = _note_content(runner)
    monkeypatch.setattr(
        run_leg, "_http_json",
        _fake_http_json_for_probe(proactive_anchor_present=False, session_start_notes=content),
    )
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    runner.probe()  # must not raise
    assert runner.record["planted_note_reachable_preflight"] is True
    preflight = runner.record["preflight"]
    assert preflight["reachable_channel"] == "session_start"
    assert preflight["proactive_probe_diagnostic_only"] is True
    assert preflight["session_start_channel"]["reachable"] is True
    # the proactive result is still recorded (diagnostic), just not authoritative
    assert preflight["turn1_text_only"]["planted_present"] is False
    assert preflight["turn2_with_file_anchor"]["planted_present"] is False


def test_probe_hook_arm_aborts_when_session_start_channel_empty_even_if_proactive_present(tmp_path, monkeypatch):
    """The inverse: proactive channel WOULD pass (anchor present there), but
    the session-start channel -- the one this arm actually ships through --
    is empty. `probe()` must still abort: proactive presence is not a
    substitute for session-start reachability on a hook arm."""
    runner = _make_hook_runner(tmp_path, arm="hook-full")
    monkeypatch.setattr(
        run_leg, "_http_json",
        _fake_http_json_for_probe(proactive_anchor_present=True, session_start_notes=None),
    )
    with pytest.raises(SystemExit, match=r"SessionStart channel"):
        runner.probe()
    assert runner.record["planted_note_reachable_preflight"] is False
    assert runner.record["preflight"]["reachable_channel"] == "session_start"


def test_probe_hook_arm_allow_unreachable_bypasses_session_start_abort(tmp_path, monkeypatch):
    runner = _make_hook_runner(tmp_path, arm="hook-sessionstart", allow_unreachable=True)
    monkeypatch.setattr(
        run_leg, "_http_json",
        _fake_http_json_for_probe(proactive_anchor_present=False, session_start_notes=None),
    )
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    runner.probe()  # must not raise
    assert runner.record["planted_note_reachable_preflight"] is False


def test_probe_non_hook_arm_still_uses_proactive_channel_unchanged(tmp_path, monkeypatch):
    """UPG-EVAL-PLANT-DISPLACEMENT: a non-hook arm's reachability gate is now
    a direct by-id `/v1/recall` integrity check, not the proactive channel's
    ranking -- see DESIGN.md 4.1's "Pre-spend reachability is a by-id
    integrity check" paragraph. This exercises the genuine-absence case (the
    daemon reports the note missing by id): it must still abort, with the
    new by-id abort message and `reachable_channel == "by_id"`."""
    runner = _make_hook_runner(tmp_path, arm="proxy")  # non-hook arm

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            return {"item_count": 0, "context": "", "anchor_ids": []}
        if method == "POST" and url.endswith("/v1/recall"):
            assert payload.get("note_id") == 7
            return {"notes": "Note #7 not found."}
        raise AssertionError(f"unexpected call in probe(): {method} {url}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    with pytest.raises(SystemExit) as exc_info:
        runner.probe()
    assert str(exc_info.value) == (
        "ABORT: the planted note is absent or content-corrupt in the "
        "daemon's store (by-id /v1/recall integrity check failed), so "
        "this leg cannot test its memory channel. Fix the scenario "
        "(not the score); re-run with --allow-unreachable to record "
        "it anyway."
    )
    assert runner.record["preflight"]["reachable_channel"] == "by_id"
    assert runner.record["preflight"]["proactive_probe_diagnostic_only"] is False
    assert runner.record["channel_delivery"] == "unreachable"


def test_probe_non_hook_arm_succeeds_via_proactive_channel_unchanged(tmp_path, monkeypatch):
    """UPG-EVAL-PLANT-DISPLACEMENT: the by-id integrity check is now the gate
    (`reachable_channel == "by_id"`); the proactive channel delivering the
    anchor at the default budget records `channel_delivery ==
    "delivered_default"` as a diagnostic, not the gate itself."""
    runner = _make_hook_runner(tmp_path, arm="mcp")  # non-hook memory arm
    content = _note_content(runner)

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            return {"item_count": 1, "context": "", "anchor_ids": ["note:7"]}
        if method == "POST" and url.endswith("/v1/recall"):
            if payload.get("note_id") == 7:
                # the new by-id integrity probe (UPG-EVAL-PLANT-DISPLACEMENT)
                return {"notes": f"[7] [HIGH] [agent]  (0m ago)\n{content}"}
            if "query" in payload:
                # arm "mcp"'s own separate query-shaped recall probe (unrelated to
                # DEFECT 7 -- pre-existing behavior for arms "mcp"/"mcp-bare")
                return {"notes": "[#7] some note"}
        raise AssertionError(f"unexpected call: {method} {url} {payload!r}")

    def fake_with_headers(method, url, payload=None, *, timeout=30.0, extra_headers=None):
        # DEFECT 13: `probe()` now also runs a pre-session MCP handshake
        # (`_mcp_handshake_probe`) against `/mcp` for arms "mcp"/"mcp-bare",
        # over the header-returning transport seam (`_http_json_with_headers`).
        assert method == "POST" and url.endswith("/mcp")
        if payload and payload.get("method") == "initialize":
            return {"jsonrpc": "2.0", "id": 1, "result": {}}, {"Mcp-Session-Id": "sess-1"}
        if payload and payload.get("method") == "tools/list":
            return {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "vectr_recall"}]}}, {}
        raise AssertionError(f"unexpected /mcp payload: {payload}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(run_leg, "_http_json_with_headers", fake_with_headers)
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    runner.probe()  # must not raise
    assert runner.record["planted_note_reachable_preflight"] is True
    assert runner.record["preflight"]["reachable_channel"] == "by_id"
    assert runner.record["channel_delivery"] == "delivered_default"
    assert runner.record["mcp_handshake_ok"] is True
    assert runner.record["mcp_handshake_tools"] == 1


# ---------------------------------------------------------------------------
# UPG-EVAL-PLANT-DISPLACEMENT (2026-08-03 ruling, DESIGN.md 4.1): a live k>=2
# leg aborted pre-spend because the agent's OWN note from a prior leg (an
# anchored, kind="gotcha", priority="high" note) legitimately outranked the
# unanchored planted directive at the proactive channel's default item
# budget. That ranking is the product working as intended -- so `probe()`
# now gates non-hook memory arms on a direct by-id `/v1/recall` integrity
# check (`_note_by_id_probe`), and treats "present but outranked" as a
# measured property of the run (`channel_delivery="displaced"`, no abort),
# reserving ABORT for genuine absence/corruption or transport-level failure.
# ---------------------------------------------------------------------------


def test_probe_displacement_records_rank_and_runs_without_abort(tmp_path, monkeypatch):
    """(a) Displacement path: the by-id integrity check passes, but the
    planted note (note:7) is absent from the proactive channel's
    default-budget response (turn1/turn2) because a stronger agent-authored
    note (note:2, kind=gotcha, priority=high, anchored) outranks it. The
    cooldown-ledger rank probe must find the planted note at rank 2, and the
    leg must RUN -- no abort -- recording the new displacement fields."""
    runner = _make_hook_runner(tmp_path, arm="proxy")
    runner.notes_count_at_start = 2  # cap = min(2, 10) = 2 rank-probe rounds available
    content = _note_content(runner)

    rankprobe_round = {"n": 0}

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            session_id = (payload or {}).get("session_id", "")
            if session_id.endswith("-turn1") or session_id.endswith("-turn2"):
                # default budget (items=1): the displacing note wins the only
                # slot, the planted note is absent
                return {
                    "item_count": 1,
                    "context": "note #2 (gotcha, agent, anchored to app/foo.py): a strong note",
                    "anchor_ids": ["note:2"],
                }
            if session_id.endswith("-rankprobe"):
                rankprobe_round["n"] += 1
                if rankprobe_round["n"] == 1:
                    # round 1: same displacing note (not yet cooldown-suppressed
                    # on this synthetic session)
                    return {
                        "item_count": 1,
                        "context": "note #2 (gotcha, agent, anchored to app/foo.py): a strong note",
                        "anchor_ids": ["note:2"],
                    }
                # round 2: note:2 is now cooldown-suppressed for this session_id
                # (the product's own cooldown ledger), so the planted note surfaces
                return {
                    "item_count": 1,
                    "context": "note #7 (gotcha, agent): the planted note",
                    "anchor_ids": ["note:7"],
                }
            raise AssertionError(f"unexpected /v1/proactive session_id: {session_id!r}")
        if method == "POST" and url.endswith("/v1/recall"):
            if payload.get("note_id") == 7:
                return {"notes": f"[7] [MEDIUM] [agent]  (0m ago)\n{content}"}
            if payload.get("note_id") == 2:
                return {"notes": "[2] [HIGH] [GOTCHA] [agent]  (0m ago)\na strong note"}
        raise AssertionError(f"unexpected call: {method} {url} {payload!r}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    runner.probe()  # must NOT raise -- displacement is not an abort

    assert runner.record["planted_note_reachable_preflight"] is True
    assert runner.record["channel_delivery"] == "displaced"
    assert runner.record["delivered_at_default"] is False
    assert runner.record["planted_rank"] == 2
    assert runner.record["displaced_by"] == [
        {"note_id": 2, "kind": "gotcha", "priority": "high", "anchored": True},
    ]
    assert runner.record["preflight"]["rank_probe"]["rank"] == 2


def test_probe_content_corrupt_by_id_still_aborts(tmp_path, monkeypatch):
    """(b) Genuine-unreachability case: the note exists at that id but its
    content does not match what was planted (content-corrupt) -- this must
    still ABORT, distinct from the accepted "displaced" case above."""
    runner = _make_hook_runner(tmp_path, arm="proxy")

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            return {"item_count": 0, "context": "", "anchor_ids": []}
        if method == "POST" and url.endswith("/v1/recall") and payload.get("note_id") == 7:
            return {"notes": "[7] [MEDIUM] [agent]  (0m ago)\nsome unrelated content that was never planted"}
        raise AssertionError(f"unexpected call: {method} {url} {payload!r}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    with pytest.raises(SystemExit) as exc_info:
        runner.probe()
    assert "absent or content-corrupt" in str(exc_info.value)
    assert runner.record["preflight"]["by_id_probe"]["exists"] is True
    assert runner.record["preflight"]["by_id_probe"]["content_matches"] is False
    assert runner.record["channel_delivery"] == "unreachable"
    assert runner.record["planted_note_reachable_preflight"] is False


def test_probe_infra_unreachable_aborts_with_infra_message(tmp_path, monkeypatch):
    """(c) Every probe path fails at the transport level (daemon unreachable /
    embedder never warms) -- must abort with the infra_unreachable message,
    distinct from the by-id-absent message in (b) above."""
    runner = _make_hook_runner(tmp_path, arm="proxy")

    def fake(method, url, payload=None, timeout=30):
        # a transport-level failure shape for every call, matching
        # `_transport_ok`'s own definition (any of "_error"/"_http_error"/"_raw")
        return {"_error": "URLError: [Errno 61] Connection refused"}

    monkeypatch.setattr(run_leg, "_http_json", fake)
    with pytest.raises(SystemExit) as exc_info:
        runner.probe()
    assert "infra unreachable" in str(exc_info.value)
    assert runner.record["preflight"]["infra_unreachable"] is True
    assert runner.record["channel_delivery"] == "unreachable"


# ---------------------------------------------------------------------------
# Arm "hook-userpromptsubmit" (D2', DESIGN.md 4, user decision 2026-08-03):
# same DEFECT-7 channel-matched reachability pattern as hook-sessionstart/
# hook-full above, but against the UserPromptSubmit channel instead of the
# SessionStart boot channel -- `_user_prompt_submit_probe()` sends the exact
# `{"query": ..., "hook_event": "UserPromptSubmit", "events":
# ["prompt-submit"]}` payload `agent/hook_cli.py::run_hook`'s own
# "user-prompt-submit" branch sends. Never `_proactive_probe`: every non-
# "proxy" arm passes `--no-inject` to the scratch proxy (`start_proxy()`),
# so a note visible to `/v1/proactive` in principle never actually ships
# through that channel during this arm's real session either.
# ---------------------------------------------------------------------------


def _fake_http_json_for_usps_probe(*, proactive_anchor_present: bool, user_prompt_submit_notes: str | None):
    """`_fake_http_json_for_probe`'s counterpart for arm "hook-userpromptsubmit":
    dispatches the same `/v1/proactive` shape, but asserts the `/v1/recall`
    call carries `hook_event="UserPromptSubmit"` + `events=["prompt-submit"]`
    (never `boot=True`) -- the payload shape unique to this arm's channel."""

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            anchor_ids = ["note:7"] if proactive_anchor_present else []
            return {"item_count": len(anchor_ids), "context": "", "anchor_ids": anchor_ids}
        if method == "POST" and url.endswith("/v1/recall"):
            assert (
                payload.get("hook_event") == "UserPromptSubmit"
                and payload.get("events") == ["prompt-submit"]
                and "query" in payload
                and payload.get("boot") is not True
            ), (
                f"probe()'s user-prompt-submit channel call must send the exact "
                f"agent/hook_cli.py::run_hook 'user-prompt-submit' branch payload "
                f"shape, got {payload!r}"
            )
            return {"notes": user_prompt_submit_notes or ""}
        raise AssertionError(f"unexpected call in probe(): {method} {url}")

    return fake


def test_probe_usps_hook_arm_judges_reachability_via_user_prompt_submit_channel_not_proactive(tmp_path, monkeypatch):
    """The proactive channel is empty but the user-prompt-submit channel
    carries the planted content -- `probe()` must NOT abort, and must record
    that it judged the user_prompt_submit channel, not proactive."""
    runner = _make_hook_runner(tmp_path, arm="hook-userpromptsubmit")
    content = _note_content(runner)
    monkeypatch.setattr(
        run_leg, "_http_json",
        _fake_http_json_for_usps_probe(proactive_anchor_present=False, user_prompt_submit_notes=content),
    )
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    runner.probe()  # must not raise
    assert runner.record["planted_note_reachable_preflight"] is True
    preflight = runner.record["preflight"]
    assert preflight["reachable_channel"] == "user_prompt_submit"
    assert preflight["proactive_probe_diagnostic_only"] is True
    assert preflight["user_prompt_submit_channel"]["reachable"] is True
    assert preflight["turn1_text_only"]["planted_present"] is False
    assert preflight["turn2_with_file_anchor"]["planted_present"] is False


def test_probe_usps_hook_arm_aborts_when_user_prompt_submit_channel_empty_even_if_proactive_present(tmp_path, monkeypatch):
    """The inverse: proactive channel WOULD pass, but the user-prompt-submit
    channel -- the one this arm actually ships through -- is empty. `probe()`
    must still abort, naming the UserPromptSubmit channel."""
    runner = _make_hook_runner(tmp_path, arm="hook-userpromptsubmit")
    monkeypatch.setattr(
        run_leg, "_http_json",
        _fake_http_json_for_usps_probe(proactive_anchor_present=True, user_prompt_submit_notes=None),
    )
    with pytest.raises(SystemExit, match=r"UserPromptSubmit channel"):
        runner.probe()
    assert runner.record["planted_note_reachable_preflight"] is False
    assert runner.record["preflight"]["reachable_channel"] == "user_prompt_submit"


# ---------------------------------------------------------------------------
# DEFECT 13: `_mcp_handshake_probe` / `_poll_recall_probe` unit coverage, plus
# `probe()` integration for the handshake ABORT-before-spend criterion.
#
# Live symptom (sentinel-verified, mcp-arm k=2): headless `claude -p` emits
# `system.init` before its async http-type MCP connect completes (a
# server can legitimately read "pending" there while still connecting), and
# current Claude Code versions never surface `mcp__vectr__*` tool schemas in
# `system.init.tools` at all -- both a compliant `initialize`/`tools/list`
# handshake straight against the daemon's own `/mcp` route (this probe) and
# transcript-level tool-use evidence (scorer.py's `_mcp_tool_use_evidence`)
# are needed as independent, non-`system.init`-dependent confirmations that
# the vectr tool surface a leg is about to depend on is actually live.
# ---------------------------------------------------------------------------


def test_mcp_handshake_probe_ok_true_on_successful_initialize_and_tools_list(monkeypatch):
    def fake_with_headers(method, url, payload=None, *, timeout=30.0, extra_headers=None):
        assert url == "http://127.0.0.1:8899/mcp"
        if payload["method"] == "initialize":
            assert extra_headers is None
            return {"jsonrpc": "2.0", "id": 1, "result": {}}, {"Mcp-Session-Id": "sess-42"}
        assert payload["method"] == "tools/list"
        assert extra_headers == {"Mcp-Session-Id": "sess-42"}
        return (
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": [
                {"name": "vectr_search"}, {"name": "vectr_recall"}, {"name": "other_tool"},
            ]}},
            {},
        )

    monkeypatch.setattr(run_leg, "_http_json_with_headers", fake_with_headers)
    out = run_leg._mcp_handshake_probe(8899)
    assert out["ok"] is True
    assert out["tool_count"] == 2  # only the two "vectr_"-prefixed names count
    assert out["session_id"] == "sess-42"


def test_mcp_handshake_probe_ok_false_when_no_vectr_prefixed_tool_present(monkeypatch):
    def fake_with_headers(method, url, payload=None, *, timeout=30.0, extra_headers=None):
        if payload["method"] == "initialize":
            return {"jsonrpc": "2.0", "id": 1, "result": {}}, {"Mcp-Session-Id": "sess-1"}
        return {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "unrelated_tool"}]}}, {}

    monkeypatch.setattr(run_leg, "_http_json_with_headers", fake_with_headers)
    out = run_leg._mcp_handshake_probe(8899)
    assert out["ok"] is False
    assert out["tool_count"] == 0


def test_mcp_handshake_probe_ok_false_on_transport_error(monkeypatch):
    monkeypatch.setattr(
        run_leg, "_http_json_with_headers",
        lambda *a, **k: ({"_error": "URLError"}, {}),
    )
    out = run_leg._mcp_handshake_probe(8899)
    assert out["ok"] is False
    assert out["session_id"] is None


def test_mcp_handshake_probe_ok_false_on_json_rpc_error_response(monkeypatch):
    def fake_with_headers(method, url, payload=None, *, timeout=30.0, extra_headers=None):
        if payload["method"] == "initialize":
            return {"jsonrpc": "2.0", "id": 1, "result": {}}, {"Mcp-Session-Id": "sess-1"}
        return {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "no such method"}}, {}

    monkeypatch.setattr(run_leg, "_http_json_with_headers", fake_with_headers)
    out = run_leg._mcp_handshake_probe(8899)
    assert out["ok"] is False


def test_poll_recall_probe_stops_early_when_note_found(monkeypatch):
    responses = [
        {"notes": "", "method": "sql"},
        {"notes": "[#9] some delivered content", "method": "semantic"},
    ]
    calls: list[dict] = []

    def fake(method, url, payload=None, timeout=30):
        assert url.endswith("/v1/recall")
        calls.append(payload)
        return responses[len(calls) - 1]

    sleeps: list[float] = []
    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(run_leg.time, "sleep", lambda s: sleeps.append(s))

    out = run_leg._poll_recall_probe(8899, "some query", 9, interval_s=2.0, timeout_s=30.0)
    assert out["returned"] is True
    assert out["method"] == "semantic"
    assert len(calls) == 2  # stopped as soon as the note was found, not all 15 polls
    assert sleeps == [2.0]  # exactly one sleep between the miss and the hit


def test_poll_recall_probe_times_out_and_records_last_method_when_note_absent(monkeypatch):
    calls = {"n": 0}

    def fake(method, url, payload=None, timeout=30):
        calls["n"] += 1
        return {"notes": "", "method": "sql"}

    clock = [0.0]
    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(run_leg.time, "time", lambda: clock[0])
    monkeypatch.setattr(run_leg.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    out = run_leg._poll_recall_probe(8899, "q", 9, interval_s=2.0, timeout_s=30.0)
    assert out["returned"] is False
    assert out["method"] == "sql"
    assert out["elapsed_s"] >= 30.0
    assert calls["n"] == 16  # polled every 2s up to and including the 30s deadline


def test_poll_recall_probe_note_id_none_returns_none_never_true(monkeypatch):
    """T0/single-shot debugging calls (no planted note) must not report a
    false "returned=True" just because the recall response happens to be
    non-empty; mirrors `leg_non_vacuity`'s own `note_id is None` skip rule."""
    monkeypatch.setattr(run_leg, "_http_json", lambda *a, **k: {"notes": "[#3] unrelated", "method": "sql"})
    monkeypatch.setattr(run_leg.time, "sleep", lambda s: None)
    out = run_leg._poll_recall_probe(8899, "q", None, interval_s=2.0, timeout_s=2.0)
    assert out["returned"] is None


def test_probe_mcp_arm_aborts_when_mcp_handshake_fails(tmp_path, monkeypatch):
    runner = _make_hook_runner(tmp_path, arm="mcp-bare")
    content = _note_content(runner)

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            return {"item_count": 1, "context": "", "anchor_ids": ["note:7"]}
        if method == "POST" and url.endswith("/v1/recall") and payload.get("note_id") == 7:
            # by-id integrity probe (UPG-EVAL-PLANT-DISPLACEMENT) runs before
            # the handshake check this test exercises, and must pass so the
            # handshake abort -- not a by-id abort -- is what's observed here.
            return {"notes": f"[7] [HIGH] [agent]  (0m ago)\n{content}"}
        raise AssertionError(f"unexpected call: {method} {url} {payload!r}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(
        run_leg, "_http_json_with_headers", lambda *a, **k: ({"_error": "URLError"}, {}),
    )
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)

    with pytest.raises(SystemExit) as exc_info:
        runner.probe()
    assert "ABORT: the MCP streamable-HTTP handshake" in str(exc_info.value)
    assert runner.record["mcp_handshake_ok"] is False
    assert runner.record["mcp_handshake_tools"] == 0
    assert (runner.artifacts / "mcp-handshake.json").exists()


def test_probe_mcp_arm_allow_unreachable_bypasses_handshake_abort(tmp_path, monkeypatch):
    runner = _make_hook_runner(tmp_path, arm="mcp", allow_unreachable=True)
    content = _note_content(runner)

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            return {"item_count": 1, "context": "", "anchor_ids": ["note:7"]}
        if method == "POST" and url.endswith("/v1/recall") and payload.get("note_id") == 7:
            return {"notes": f"[7] [HIGH] [agent]  (0m ago)\n{content}"}
        raise AssertionError(f"unexpected call: {method} {url} {payload!r}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(
        run_leg, "_http_json_with_headers", lambda *a, **k: ({"_error": "URLError"}, {}),
    )
    monkeypatch.setattr(
        run_leg, "_poll_recall_probe",
        lambda *a, **k: {"returned": False, "elapsed_s": 30.0, "method": "sql"},
    )
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)

    runner.probe()  # handshake failed, but --allow-unreachable must suppress the abort
    assert runner.record["mcp_handshake_ok"] is False
    assert runner.record["recall_probe_returned_note"] is False
    assert runner.record["recall_probe_method"] == "sql"
    assert runner.record["recall_probe_elapsed_s"] == 30.0


# ---------------------------------------------------------------------------
# DEFECT 8: whitespace-normalized (and, where the haystack is raw JSON text,
# JSON-escape-aware) delivery-containment checks.
#
# Live T2 symptom: `release_via_ci-proxy-verifiable-s0` k2 (arm "proxy") came
# back invalid with "T3: variant's provenance-trail text absent from probe's
# returned context", even though `preflight.json` showed `planted_present=
# true` -- the note WAS delivered. Root cause: the S1-verifiable NoteVariant's
# trail has an internal newline (mirrored here by `bench_box_only`'s real
# "provenance" variant, which also has one -- `_S6_FACT + "\nEstablished
# ..."`), but the proactive/hook delivery channels collapse ALL whitespace on
# every injected note (`agent/proactive/matcher.py::_one_line`, deliberate
# product behavior -- do not "fix" the product). `probe()`'s old trail check
# did a literal `trail_text in context` against the UN-collapsed trail, which
# is structurally False for any multi-line trail regardless of delivery.
#
# `hook_preflight()`'s stdout check and `scorer.leg_non_vacuity`'s D1
# transcript check have a COMPOUND version of the same symptom: their
# haystack is raw JSON text (`print(json.dumps({"hookSpecificOutput": {...,
# "additionalContext": text}}))` / a stream-json transcript file), so a real
# newline in the source content is JSON-escaped there to the two literal
# characters `\n` -- whitespace-collapsing the haystack alone cannot match
# that back to a real newline, since `\` and `n` are not whitespace.
# `_content_delivered_in_json_text` checks both the collapsed literal form
# and the collapsed JSON-string-escaped form of the content.
# ---------------------------------------------------------------------------


def test_collapse_ws_matches_agent_proactive_matcher_one_line_semantics():
    """`_collapse_ws` must mirror `agent/proactive/matcher.py::_one_line`
    (`" ".join(text.split())`) exactly -- same collapsing of runs of any
    whitespace (space, tab, newline) to a single space, leading/trailing
    stripped."""
    multiline = "  Established 2026-07-12\n(no API token exists for this\tproject).  "
    assert run_leg._collapse_ws(multiline) == (
        "Established 2026-07-12 (no API token exists for this project)."
    )


def test_probe_trail_text_delivered_true_for_multiline_trail_via_whitespace_collapse(tmp_path, monkeypatch):
    """Reproduces the live T3 failure directly (`release_via_ci-proxy-verifiable-s0`
    k2): `release_via_ci`'s real 'verifiable' NoteVariant content has an internal
    newline in its trail (between the "...project)." sentence and the "Verify:
    grep..." line -- survives `.replace(fact_sentence, "", 1).strip()` because it
    sits strictly between two authored segments, not at a boundary `.strip()`
    would remove). The proactive channel delivers it whitespace-collapsed (as the
    product always does); this must be recognized as delivered. Was red under the
    pre-DEFECT-8 code (`trail_text in context` against the un-collapsed trail)."""
    runner = _make_hook_runner(tmp_path, arm="proxy", scenario="release_via_ci", note_variant="verifiable")
    variant = runner._find_note_variant()
    trail_text = variant.content.replace(runner.scenario.fact_sentence, "", 1).strip()
    assert "\n" in trail_text, "fixture sanity: this variant's trail must be multi-line"
    delivered_context = run_leg._collapse_ws(trail_text)  # exactly what the channel delivers

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            return {"item_count": 1, "context": delivered_context, "anchor_ids": ["note:7"]}
        if method == "POST" and url.endswith("/v1/recall") and payload.get("note_id") == 7:
            # by-id integrity probe (UPG-EVAL-PLANT-DISPLACEMENT)
            return {"notes": f"[7] [HIGH] [agent]  (0m ago)\n{variant.content}"}
        raise AssertionError(f"unexpected call: {method} {url} {payload!r}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    runner.probe()  # must not raise (anchor present -> reachable)
    assert runner.trail_text_delivered is True
    assert runner.record["trail_text_delivered"] is True


def test_probe_trail_text_delivered_false_when_genuinely_absent(tmp_path, monkeypatch):
    """A multi-line trail that truly never appears in either probe turn's
    context must still read as not-delivered -- whitespace normalization
    must not manufacture a false positive."""
    runner = _make_hook_runner(tmp_path, arm="proxy", scenario="release_via_ci", note_variant="verifiable")
    content = _note_content(runner)

    def fake(method, url, payload=None, timeout=30):
        if method == "POST" and url.endswith("/v1/proactive"):
            # anchor present (so probe() doesn't abort on reachability) but the
            # rendered context text is unrelated to the trail
            return {"item_count": 1, "context": "totally unrelated context text", "anchor_ids": ["note:7"]}
        if method == "POST" and url.endswith("/v1/recall") and payload.get("note_id") == 7:
            # by-id integrity probe (UPG-EVAL-PLANT-DISPLACEMENT)
            return {"notes": f"[7] [HIGH] [agent]  (0m ago)\n{content}"}
        raise AssertionError(f"unexpected call: {method} {url} {payload!r}")

    monkeypatch.setattr(run_leg, "_http_json", fake)
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    runner.probe()  # must not raise -- reachable via anchor; trail delivery is a separate diagnostic
    assert runner.trail_text_delivered is False
    assert runner.record["trail_text_delivered"] is False


def test_content_delivered_in_json_text_matches_multiline_content_json_escaped_in_stdout():
    """Unit-level proof of `_content_delivered_in_json_text` against exactly
    the shape `agent/hook_cli.py::_emit_hook_context` produces
    (`print(json.dumps({"hookSpecificOutput": {..., "additionalContext":
    text}}))`): a real newline in `content` is JSON-escaped to `\\n` in
    `stdout`, so plain (even whitespace-collapsed) containment fails, but
    this helper must still find it."""
    content = (
        "Established 2026-07-12 (no API token exists for this project).\n"
        'Verify: grep -n "id-token" .github/workflows/release.yml'
    )
    stdout = json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": f"# Triggered Memory (1 fired)\n\n[7] [DIRECTIVE]\n  {content}",
        }
    })
    # sanity: the bug this helper fixes is real for this fixture
    assert content not in stdout
    assert run_leg._collapse_ws(content) not in run_leg._collapse_ws(stdout)
    assert run_leg._content_delivered_in_json_text(content, stdout) is True
    assert run_leg._content_delivered_in_json_text("never delivered anywhere", stdout) is False
    assert run_leg._content_delivered_in_json_text("", stdout) is False


def test_hook_preflight_stdout_content_check_matches_multiline_note_json_escaped(tmp_path, monkeypatch):
    """End-to-end (within `hook_preflight()`) version of the unit test above:
    a multi-line planted note's content, JSON-escaped inside the hook
    subprocess's real stdout shape, must be recognized as delivered and must
    not abort the preflight."""
    runner = _make_hook_runner(tmp_path, arm="hook-sessionstart", scenario="release_via_ci", note_variant="verifiable")
    content = _note_content(runner)
    assert "\n" in content, "fixture sanity: this variant's content must be multi-line"

    status_calls = {"n": 0}

    def fake_http_json(method, url, payload=None, timeout=30):
        status_calls["n"] += 1
        count = 0 if status_calls["n"] == 1 else 1
        return {"hook_injection_counts": {"SessionStart": count}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        stdout = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": f"# Triggered Memory (1 fired)\n\n[7] [DIRECTIVE]\n  {content}",
            }
        })
        return 0, stdout, "", False

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)

    runner.hook_preflight()  # must not raise

    hp = runner.record["hook_preflight"]
    assert hp["daemon_evidence"] is True
    assert hp["stdout_has_planted_content"] is True


# ---------------------------------------------------------------------------
# DEFECT 10 (direction 1, user decision 2026-07-30; DESIGN.md 6.5): legs k=1..3 of
# a trajectory share one persistent workspace, so a scenario-critical artifact one
# leg's compliant task completes (e.g. S5's `deploy/queue.yaml` staging entry) can
# make a LATER leg's own check vacuous -- every arm already sees the entry present
# and the check can no longer tell "this leg complied" from "leg k-1 already did".
# `LegRunner._apply_critical_residue_reset()` restores exactly the paths a scenario
# declares in `critical_residue_paths` to their scenario-seed content, at the start
# of every k>=2 leg, AFTER `_restore_and_verify`'s manifest integrity check (so
# DEFECT 9-adjacent tar-fidelity verification is unaffected) and before the agent
# runs. Every other file -- declared or not -- keeps its natural cross-leg residue.
# ---------------------------------------------------------------------------


def _s5_leg1_end_state_tar(tmp_path: Path) -> tuple[Path, str]:
    """Builds a restore-tar simulating S5's REAL leg-1 end state: the scenario seed
    plus leg 1's own compliant addition to `deploy/queue.yaml` (its
    fact_acquisition target) -- exactly the residue DEFECT 10 exists to neutralize
    -- plus a mutation to `deploy/README.md`, a file no leg ever legitimately
    writes but which this fixture perturbs anyway to prove the reset touches ONLY
    the declared path, not "everything under deploy/". Returns (tar_path,
    manifest_sha256), computed the identical way `LegRunner.snapshot()` computes a
    real leg's own end-state manifest.
    """
    s5 = scen.SCENARIOS["deploy_reverted_by_reconciler"]
    leg1_ws = tmp_path / "leg1-end-state"
    scen.materialize(s5, leg1_ws)

    queue = leg1_ws / "deploy/queue.yaml"
    queue.write_text(
        queue.read_text(encoding="utf-8") + '- date: "2026-07-16"\n  target: staging\n  ref: HEAD\n',
        encoding="utf-8",
    )
    readme = leg1_ws / "deploy/README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "Also see the release-bot runbook.\n", encoding="utf-8"
    )

    baselines = run_leg._sha256_tree(leg1_ws)
    manifest_hash = run_leg._write_manifest(baselines, tmp_path / "leg1-manifest.sha256")
    tar_path = tmp_path / "leg1-end-state.tar"
    run_leg._make_tar(leg1_ws, tar_path)
    return tar_path, manifest_hash


def test_restore_and_verify_resets_declared_path_leaves_undeclared_residue_untouched(tmp_path):
    """The core acceptance case: reconstructs the k=2 leg-start state from a
    faithful leg-1 end-state tar and shows the fix -- `deploy/queue.yaml` (declared)
    comes back to the scenario seed, while `deploy/README.md` (real, un-declared
    residue) keeps leg 1's edit untouched.
    """
    tar_path, manifest_hash = _s5_leg1_end_state_tar(tmp_path)
    runner = _make_runner(
        tmp_path,
        scenario="deploy_reverted_by_reconciler",
        restore_tar=str(tar_path),
        restore_manifest_sha256=manifest_hash,
    )
    runner.artifacts.mkdir(parents=True, exist_ok=True)
    s5 = scen.SCENARIOS["deploy_reverted_by_reconciler"]

    # Sanity: before the fix runs, the extracted tar really does carry leg 1's
    # residue (i.e. the fixture reproduces the pre-fix symptom) -- proven directly
    # against a plain extract further down via the no-op comparison test.
    runner._restore_and_verify()

    queue_content = (runner.workspace / "deploy/queue.yaml").read_text(encoding="utf-8")
    assert queue_content == s5.files["deploy/queue.yaml"], (
        "declared critical_residue_paths entry must be restored to scenario seed, "
        "not left carrying leg 1's staging entry"
    )
    readme_content = (runner.workspace / "deploy/README.md").read_text(encoding="utf-8")
    assert "Also see the release-bot runbook." in readme_content, (
        "non-declared residue must persist untouched -- only declared paths reset"
    )

    # leg_start_baselines (and therefore baselines.json / every FileUnchanged /
    # FileMutated check this leg runs) reflect the POST-reset tree, not the raw
    # restore -- scorer.py's evaluate_check contract (baselines are always this
    # leg's true start-of-leg state).
    recomputed = run_leg._sha256_tree(runner.workspace, extra_ignore=runner._extra_ignore_paths())
    assert runner.leg_start_baselines == recomputed
    assert runner.record["critical_residue_reset_paths"] == ["deploy/queue.yaml"]


def test_apply_critical_residue_reset_is_a_noop_when_scenario_declares_no_paths(tmp_path):
    """Reproduces the pre-fix symptom directly: a scenario shaped like S5 was
    BEFORE this fix (no `critical_residue_paths` declared) leaves leg 1's residue
    in place at leg 2's start -- `_apply_critical_residue_reset` is a true
    structural no-op when nothing is declared, not special-cased to S5's slug.
    """
    tar_path, manifest_hash = _s5_leg1_end_state_tar(tmp_path)
    runner = _make_runner(
        tmp_path,
        scenario="deploy_reverted_by_reconciler",
        restore_tar=str(tar_path),
        restore_manifest_sha256=manifest_hash,
    )
    runner.artifacts.mkdir(parents=True, exist_ok=True)
    runner.scenario = dataclasses.replace(runner.scenario, critical_residue_paths=())

    runner._restore_and_verify()

    content = (runner.workspace / "deploy/queue.yaml").read_text(encoding="utf-8")
    assert "target: staging" in content, (
        "pre-fix bug reproduced: leg 1's staging entry is still present at leg 2's "
        "start when nothing is declared for reset"
    )
    assert "critical_residue_reset_paths" not in runner.record


def test_critical_residue_reset_runs_after_manifest_integrity_check_not_before(tmp_path):
    """Ordering guarantee: a genuinely drifted restore (manifest mismatch) still
    aborts, exactly as before this fix -- the reset must never mask real tar
    corruption/drift by running ahead of the integrity check.
    """
    tar_path, _correct_hash = _s5_leg1_end_state_tar(tmp_path)
    runner = _make_runner(
        tmp_path,
        scenario="deploy_reverted_by_reconciler",
        restore_tar=str(tar_path),
        restore_manifest_sha256="0" * 64,  # deliberately wrong
    )
    runner.artifacts.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit, match="manifest mismatch"):
        runner._restore_and_verify()
    # The abort fires before any reset bookkeeping is recorded.
    assert "critical_residue_reset_paths" not in runner.record


def test_critical_residue_reset_is_a_noop_for_a_scenario_with_no_declared_paths(tmp_path):
    """Cross-scenario parity: a scenario that legitimately declares nothing (e.g.
    `release_via_ci` -- every leg targets a distinct release artifact) is completely
    unaffected by this mechanism -- `_apply_critical_residue_reset` must not raise or
    mutate anything when `critical_residue_paths` is empty.
    """
    scenario = scen.SCENARIOS["release_via_ci"]
    assert scenario.critical_residue_paths == ()
    runner = _make_runner(tmp_path, scenario="release_via_ci", k=1)
    runner.workspace.mkdir(parents=True, exist_ok=True)
    runner.artifacts.mkdir(parents=True, exist_ok=True)
    before = dict(runner.leg_start_baselines)

    runner._apply_critical_residue_reset()

    assert runner.leg_start_baselines == before
    assert "critical_residue_reset_paths" not in runner.record


# ---------------------------------------------------------------------------
# DEFECT 11 / UPG-EVAL-PATH-SLUG-LEAK: the agent-visible cwd -- and the
# SessionStart hook's own synthetic session-init stdin payload, which
# `hook_preflight()` builds from that same cwd -- must never surface the
# scenario slug, arm, or note-variant. `run_agent()`'s real `claude -p`
# subprocess and `hook_preflight()`'s synthetic check both use `_agent_cwd()`
# ("one function so the two can never drift apart" -- its own docstring), so
# pinning this one property covers both routes at once. `run_plan.py` mints
# the opaque `--workspace-dir` this pins; these tests exercise the LegRunner
# side of that same fix using the exact resolver it now calls.
# ---------------------------------------------------------------------------


def test_agent_cwd_carries_no_scenario_arm_or_variant_tokens_for_a_fresh_s5_trajectory(tmp_path):
    spec = run_plan.TrajectorySpec("deploy_reverted_by_reconciler", "proxy", "provenance", 0, 3)
    traj_dir = run_plan.resolve_traj_dir(tmp_path, spec.trajectory_id)
    workspace_dir = run_plan._trajectory_workspace_dir(traj_dir)

    runner = _make_runner(
        tmp_path, scenario="deploy_reverted_by_reconciler", arm="proxy",
        note_variant="provenance", workspace_dir=str(workspace_dir),
    )
    cwd_str = str(runner._agent_cwd())

    assert "deploy_reverted_by_reconciler" not in cwd_str
    assert "reconciler" not in cwd_str
    assert "provenance" not in cwd_str
    # "proxy" is a common-enough fragment that it is checked as its own path
    # COMPONENT, not merely a substring, to keep this assertion precise.
    assert "proxy" not in Path(cwd_str).parts


def test_hook_preflight_session_init_payload_carries_no_scenario_or_arm_tokens(tmp_path, monkeypatch):
    """The synthetic SessionStart hook stdin payload `hook_preflight()` builds
    (`cwd`, mirroring exactly what a real Claude Code SessionStart hook
    invocation receives) is built directly from `_agent_cwd()` -- this pins the
    same property at the actual call site, not just the helper above.
    """
    spec = run_plan.TrajectorySpec("deploy_reverted_by_reconciler", "proxy", "provenance", 0, 3)
    traj_dir = run_plan.resolve_traj_dir(tmp_path, spec.trajectory_id)
    workspace_dir = run_plan._trajectory_workspace_dir(traj_dir)

    runner = _make_hook_runner(
        tmp_path, scenario="deploy_reverted_by_reconciler",
        note_variant="provenance", workspace_dir=str(workspace_dir),
    )
    content = _note_content(runner)
    captured: dict = {}
    status_calls = {"n": 0}

    def fake_http_json(method, url, payload=None, timeout=30):
        status_calls["n"] += 1
        # Before the hook call: 0 injections; after: 1 -- a real delta, matching
        # the daemon-evidence pattern the existing hook_preflight tests use.
        count = 0 if status_calls["n"] == 1 else 1
        return {"hook_injection_counts": {"SessionStart": count}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        captured["cwd"] = cwd
        captured["stdin"] = json.loads(stdin)
        stdout = json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": content}
        })
        return 0, stdout, "", False

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)

    runner.hook_preflight()

    assert "deploy_reverted_by_reconciler" not in captured["cwd"]
    assert "deploy_reverted_by_reconciler" not in captured["stdin"]["cwd"]
    assert "provenance" not in captured["stdin"]["cwd"]
    assert captured["stdin"]["cwd"] == captured["cwd"] == str(runner._agent_cwd())


# ---------------------------------------------------------------------------
# DEFECT 11 PII hygiene (user decision 2026-07-31): every leg workspace's git
# identity must be pinned to a synthetic pair so an agent's own `git commit`
# can never surface the operator's real name/email. Root cause: `materialize()`'s
# initial commit uses EPHEMERAL `-c user.email=.../-c user.name=...` flags,
# which do not persist into `.git/config` -- so without this fix, the agent's
# own later commits fell through to the operator's real global ~/.gitconfig.
# ---------------------------------------------------------------------------


def _git_config_get(repo: Path, key: str) -> str:
    return subprocess.run(
        ["git", "config", "--get", key], cwd=str(repo), capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_materialize_pins_synthetic_git_identity_for_a_git_init_scenario(tmp_path):
    s1 = scen.SCENARIOS["release_via_ci"]
    ws = tmp_path / "ws"
    scen.materialize(s1, ws)

    assert scen.SYNTHETIC_GIT_USER_NAME == "Mary Doe"
    assert scen.SYNTHETIC_GIT_USER_EMAIL == "mary.doe@example.com"
    assert _git_config_get(ws, "user.name") == scen.SYNTHETIC_GIT_USER_NAME
    assert _git_config_get(ws, "user.email") == scen.SYNTHETIC_GIT_USER_EMAIL


def test_pin_synthetic_git_identity_is_a_noop_for_a_non_git_scenario(tmp_path):
    """Most scenarios never `git_init` at all -- must not create a repo that
    isn't already there, and must not raise."""
    ws = tmp_path / "ws"
    ws.mkdir()
    scen.pin_synthetic_git_identity(ws)  # must not raise
    assert not (ws / ".git").exists()


def test_prepare_pins_synthetic_git_identity_on_restore_for_k_ge_2(tmp_path):
    """The k==1 `materialize()` path pins it internally (previous test); this
    pins the OTHER half -- `LegRunner.prepare()`'s uniform call after
    `_restore_and_verify()` -- so a trajectory can never end up mid-run with a
    repo whose local config reverted away from the synthetic identity on a
    later leg, regardless of what the tar being restored happens to carry.
    """
    s1 = scen.SCENARIOS["release_via_ci"]
    leg1_ws = tmp_path / "leg1-end-state"
    scen.materialize(s1, leg1_ws)
    # Simulate drift (e.g. an agent action) between leg 1's end state and this
    # leg's restore -- prepare() must re-pin regardless of what k=1 left behind.
    subprocess.run(["git", "config", "user.name", "someone else"], cwd=str(leg1_ws), check=True)

    baselines = run_leg._sha256_tree(leg1_ws)
    manifest_hash = run_leg._write_manifest(baselines, tmp_path / "leg1-manifest.sha256")
    tar_path = tmp_path / "leg1-end-state.tar"
    run_leg._make_tar(leg1_ws, tar_path)

    runner = _make_runner(
        tmp_path, scenario="release_via_ci", k=2,
        restore_tar=str(tar_path), restore_manifest_sha256=manifest_hash,
    )
    runner.prepare()

    assert _git_config_get(runner._agent_cwd(), "user.name") == scen.SYNTHETIC_GIT_USER_NAME
    assert _git_config_get(runner._agent_cwd(), "user.email") == scen.SYNTHETIC_GIT_USER_EMAIL


def test_spawn_env_for_agent_strips_and_resets_git_identity_env_vars(monkeypatch):
    """The env-var layer of the same defense: GIT_AUTHOR_*/GIT_COMMITTER_* take
    precedence over BOTH repo-local and global git config, so this is what
    covers a git repo the agent inits itself mid-session, with no local config
    pinned yet."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Real Operator")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "real.operator@example.org")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Real Operator")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "real.operator@example.org")

    env = run_leg._spawn_env_for_agent("http://127.0.0.1:1234")

    assert env["GIT_AUTHOR_NAME"] == scen.SYNTHETIC_GIT_USER_NAME
    assert env["GIT_AUTHOR_EMAIL"] == scen.SYNTHETIC_GIT_USER_EMAIL
    assert env["GIT_COMMITTER_NAME"] == scen.SYNTHETIC_GIT_USER_NAME
    assert env["GIT_COMMITTER_EMAIL"] == scen.SYNTHETIC_GIT_USER_EMAIL
    assert "Real Operator" not in env.values()
    assert "real.operator@example.org" not in env.values()


def test_spawn_env_for_agent_sets_git_config_global_to_synthetic_file(monkeypatch):
    """UPG-EVAL-IDENTITY-ENV-ISOLATION: GIT_AUTHOR_*/GIT_COMMITTER_* only affect
    commit authorship, never a `git config` READ, so a workspace with no `.git`
    at all (e.g. S5, where `deploy/queue.yaml`'s `requested_by` field is filled
    in by agent judgment rather than a commit) still resolves `git config
    user.email` to the operator's real `~/.gitconfig` unless GIT_CONFIG_GLOBAL
    is also redirected. Also proves any pre-existing GIT_CONFIG_GLOBAL from the
    operator's own shell is stripped, not merely shadowed."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/Users/real-operator/.gitconfig")

    env = run_leg._spawn_env_for_agent("http://127.0.0.1:1234")

    assert env["GIT_CONFIG_GLOBAL"] != "/Users/real-operator/.gitconfig"
    synthetic_path = Path(env["GIT_CONFIG_GLOBAL"])
    assert synthetic_path.is_file()
    content = synthetic_path.read_text(encoding="utf-8")
    assert scen.SYNTHETIC_GIT_USER_NAME in content
    assert scen.SYNTHETIC_GIT_USER_EMAIL in content


def test_git_config_read_in_gitless_workspace_resolves_synthetic_identity_only_with_env(tmp_path, monkeypatch):
    """End-to-end proof of the exact leak (deploy s1 shared leg1, observed
    2026-08-03): a workspace with NO `.git` directory (S5's shape) still lets
    `git config user.email` resolve through to whatever global config is active
    -- without `_spawn_env_for_agent`'s GIT_CONFIG_GLOBAL redirect that would be
    the operator's own `~/.gitconfig` (a real PII leak); with it, the synthetic
    identity, deterministically, regardless of the operator's actual global
    config content."""
    fake_real_global = tmp_path / "real-operator-gitconfig"
    fake_real_global.write_text("[user]\n\tname = Real Operator\n\temail = real.operator@example.org\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(fake_real_global))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    gitless_workspace = tmp_path / "workspace-no-git"
    gitless_workspace.mkdir()
    assert not (gitless_workspace / ".git").exists()

    # Without the harness's env (simulating the pre-fix world): the leak.
    leaked = subprocess.run(
        ["git", "config", "user.email"], cwd=str(gitless_workspace),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert leaked == "real.operator@example.org"

    # With _spawn_env_for_agent's env: the synthetic identity, every time.
    agent_env = run_leg._spawn_env_for_agent("http://127.0.0.1:1234")
    protected = subprocess.run(
        ["git", "config", "user.email"], cwd=str(gitless_workspace),
        capture_output=True, text=True, check=True, env=agent_env,
    ).stdout.strip()
    assert protected == scen.SYNTHETIC_GIT_USER_EMAIL


# ---------------------------------------------------------------------------
# UPG-EVAL-STORE-MATCH: exact-count-1 store premise for non-tool arms
# ---------------------------------------------------------------------------
#
# Root cause: `run_plan.py`'s failed-leg retry (`_supersede(leg_dir)`) only
# renames away a leg's own artifact directory, never the trajectory-persistent
# `--db-dir`. A k==2 attempt that plants successfully and then fails a LATER
# step in the same leg (arms "hook-sessionstart"/"hook-full" run
# write_hooks()/hook_preflight() after plant_note(); arm "proxy" does not) gets
# retried, and the retry plants a second note on top of the first, with no
# reset in between. Recorded runs showed hook legs starting with 2 (k2) / 3
# (k3) notes in store against exactly 1 in every proxy leg. Fix: (a)
# `plant_note()` clears the store via `/v1/memory/clear` immediately before
# planting, and (b) `verify_note_store_matched()` gates
# `notes_count_at_start == 1` pre-spend for every arm without live vectr tool
# access (arms "mcp"/"mcp-bare" are exempt: their agent can legitimately grow
# the store past 1 via its own `vectr_remember` calls in an earlier leg).


def test_verify_note_store_matched_aborts_when_count_is_not_one_for_proxy_arm(tmp_path):
    runner = _make_runner(tmp_path)
    runner.notes_count_at_start = 2
    with pytest.raises(SystemExit) as excinfo:
        runner.verify_note_store_matched()
    assert "!= 1" in str(excinfo.value)
    assert runner.record["store_count_matched"] is False


def test_verify_note_store_matched_aborts_when_count_is_zero_for_proxy_arm(tmp_path):
    runner = _make_runner(tmp_path)
    runner.notes_count_at_start = 0
    with pytest.raises(SystemExit):
        runner.verify_note_store_matched()


def test_verify_note_store_matched_passes_when_count_is_exactly_one(tmp_path):
    runner = _make_runner(tmp_path)
    runner.notes_count_at_start = 1
    runner.verify_note_store_matched()  # must not raise
    assert "store_count_matched" not in runner.record


def test_verify_note_store_matched_exempts_mcp_arm_with_count_above_one(tmp_path):
    """Arms with live vectr tool access can legitimately carry more than one
    note into a later leg (the agent's own vectr_remember calls from an
    earlier leg) -- this is real measured behavior, not the defect the
    exact-count-1 gate targets, so these arms are checked only for "> 0"
    elsewhere (scorer.leg_non_vacuity), never here.
    """
    runner = _make_runner(tmp_path, arm="mcp")
    runner.notes_count_at_start = 4
    runner.verify_note_store_matched()  # must not raise
    assert "store_count_matched" not in runner.record


def test_verify_note_store_matched_exempts_mcp_bare_arm_with_count_above_one(tmp_path):
    runner = _make_runner(tmp_path, arm="mcp-bare")
    runner.notes_count_at_start = 3
    runner.verify_note_store_matched()  # must not raise


def test_verify_note_store_matched_is_a_noop_for_arm_none(tmp_path):
    """Arm 'none' never plants (`--plant-note` is rejected outright for it in
    `_validate_args`), so there is no plant-count invariant to check here --
    its own store-emptiness premise is asserted by scorer.leg_non_vacuity's
    "none" branch instead.
    """
    runner = _make_runner(tmp_path, arm="none")
    runner.notes_count_at_start = 0
    runner.verify_note_store_matched()  # must not raise
    runner.notes_count_at_start = 5
    runner.verify_note_store_matched()  # still must not raise -- out of scope here


def test_verify_note_store_matched_checks_hook_arms_too(tmp_path):
    for arm in ("hook-sessionstart", "hook-full", "hook-userpromptsubmit"):
        runner = _make_runner(tmp_path, arm=arm)
        runner.notes_count_at_start = 2
        with pytest.raises(SystemExit):
            runner.verify_note_store_matched()
        runner2 = _make_runner(tmp_path, arm=arm)
        runner2.notes_count_at_start = 1
        runner2.verify_note_store_matched()  # must not raise


def test_verify_note_store_matched_allow_override_flag_records_but_does_not_abort(tmp_path):
    runner = _make_runner(tmp_path, allow_store_mismatch=True)
    runner.notes_count_at_start = 2
    runner.verify_note_store_matched()  # must not raise
    assert runner.record["store_count_matched"] is False
    assert runner.record["store_mismatch_allowed"] is True


def test_reset_note_store_calls_memory_clear_and_records_deleted_count(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path)

    calls: list[tuple[str, str]] = []

    def fake_http_json(method, url, payload=None, timeout=30):
        calls.append((method, url))
        return {"deleted": 5}

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)

    runner._reset_note_store()

    assert ("POST", f"http://127.0.0.1:{runner.daemon_port}/v1/memory/clear") in calls
    assert runner.record["note_store_reset_before_plant"] == 5


# ---------------------------------------------------------------------------
# UPG-EVAL-HOOKCOUNT-CUMULATIVE: `capture_and_teardown()` used to read
# end-of-leg CUMULATIVE `hook_injection_counts` straight off `/v1/status` for
# arms "hook-sessionstart"/"hook-full" -- which also carries this leg's own
# pre-agent `hook_preflight()` subprocess call and (in the real `main()` call
# order) `probe()`'s `_session_start_probe()` daemon-side reachability check,
# both of which legitimately fire the same counters before the real agent
# session (`run_agent()`) ever starts. `run_agent()` now brackets ITS OWN
# subprocess with before/after `/v1/status` snapshots for these two arms too
# (the same pattern it already used for arm "hook-userpromptsubmit"'s
# `user_prompt_submit_injection_delta`) and records the per-event DELTA as
# `record["hook_injection_counts"]`; `capture_and_teardown()` no longer
# overwrites that field with the raw end-of-leg cumulative snapshot.
# `hook_preflight()` also re-settles `self.audit_offset` after its own hook
# subprocess call runs, so the "post-offset" measured window `scorer.
# leg_non_vacuity` reads from `audit_since_offset` excludes hook_preflight's
# own traffic too, not just `probe()`'s.
# ---------------------------------------------------------------------------


def test_run_agent_records_hook_injection_counts_delta_excluding_preflight_traffic(tmp_path, monkeypatch):
    """A leg where `hook_preflight()` already bumped the daemon's SessionStart
    counter by 1 (1 -> 2, simulating `probe()`'s own earlier `1` from
    `_session_start_probe()`) before `run_agent()` ever runs, and the real
    agent session bumps it by 1 more (2 -> 3): the leg's recorded
    `hook_injection_counts` must be the AGENT-SESSION-ONLY delta
    ({"SessionStart": 1}), never the end-of-leg cumulative (3)
    `capture_and_teardown()`'s own final `/v1/status` snapshot shows.
    """
    runner = _make_hook_runner(tmp_path, arm="hook-sessionstart")
    content = _note_content(runner)

    status_responses = iter([
        {"hook_injection_counts": {"SessionStart": 1}},  # hook_preflight(): before
        {"hook_injection_counts": {"SessionStart": 2}},  # hook_preflight(): after (preflight passes)
        {"hook_injection_counts": {"SessionStart": 2}},  # run_agent(): before
        {"hook_injection_counts": {"SessionStart": 3}},  # run_agent(): after (real session fired once)
        {"hook_injection_counts": {"SessionStart": 3}},  # capture_and_teardown(): final diagnostic read
    ])

    def fake_http_json(method, url, payload=None, timeout=30):
        if method == "GET" and url.endswith("/v1/status"):
            return next(status_responses)
        if method == "GET" and url.endswith("/__vectr_proxy/health"):
            return {"metrics": {"injected": 0}}
        raise AssertionError(f"unexpected call {method} {url}")

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        stdout = json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": content}
        })
        return 0, stdout, "", False

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)
    monkeypatch.setattr(run_leg.subprocess, "Popen", lambda cmd, **kwargs: _FakeProc())
    monkeypatch.setattr(run_leg.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 0)
    monkeypatch.setattr(run_leg, "_hard_stop_port", lambda port: [])
    monkeypatch.setattr(run_leg, "_stop_daemon", lambda *a, **k: [])

    runner.hook_preflight()
    runner.run_agent()
    runner.capture_and_teardown()

    assert runner.record["hook_injection_counts"] == {"SessionStart": 1}


def test_capture_and_teardown_still_records_cumulative_final_as_diagnostic(tmp_path, monkeypatch):
    """The raw end-of-leg cumulative snapshot is still captured somewhere for
    post-mortem/diagnostic purposes -- just no longer under the
    `hook_injection_counts` key `scorer.leg_non_vacuity` actually reads."""
    runner = _make_hook_runner(tmp_path, arm="hook-full")
    runner.record["hook_injection_counts"] = {"SessionStart": 1}  # simulate run_agent()'s own delta

    def fake_http_json(method, url, payload=None, timeout=30):
        if method == "GET" and url.endswith("/v1/status"):
            return {"hook_injection_counts": {"SessionStart": 9, "UserPromptSubmit": 4}}
        if method == "GET" and url.endswith("/__vectr_proxy/health"):
            return {"metrics": {"injected": 0}}
        raise AssertionError(f"unexpected call {method} {url}")

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_hard_stop_port", lambda port: [])
    monkeypatch.setattr(run_leg, "_stop_daemon", lambda *a, **k: [])

    runner.capture_and_teardown()

    # The delta recorded by run_agent() is untouched by teardown...
    assert runner.record["hook_injection_counts"] == {"SessionStart": 1}
    # ...while the raw cumulative is still visible under a diagnostic-only key.
    assert runner.record["hook_injection_counts_cumulative_final"] == {
        "SessionStart": 9, "UserPromptSubmit": 4,
    }


def test_hook_preflight_resettles_audit_offset_so_measured_window_excludes_its_own_traffic(tmp_path, monkeypatch):
    """`probe()` (run strictly before `hook_preflight()` in `main()`'s call
    order) already settles `self.audit_offset` once, but that settle cannot
    account for traffic `hook_preflight()` itself generates afterwards.
    `hook_preflight()` must re-settle the offset on its own successful exit so
    `scorer.leg_non_vacuity`'s `audit_since_offset` genuinely starts at the
    real agent session, not somewhere inside this leg's own preflight.
    """
    runner = _make_hook_runner(tmp_path, arm="hook-sessionstart")
    content = _note_content(runner)
    runner.audit_offset = 100  # simulates probe()'s own earlier settle
    runner.record["audit_offset_after_preflight"] = 100

    status_calls = {"n": 0}

    def fake_http_json(method, url, payload=None, timeout=30):
        assert method == "GET" and url.endswith("/v1/status")
        status_calls["n"] += 1
        # Before the hook call: 0 injections; after: 1 -- a real delta, so
        # this preflight PASSES and execution reaches the re-settle code.
        count = 0 if status_calls["n"] == 1 else 1
        return {"hook_injection_counts": {"SessionStart": count}}

    def fake_run_hook_command(command, *, cwd, stdin, env, timeout=60.0):
        stdout = json.dumps({
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": content}
        })
        return 0, stdout, "", False

    monkeypatch.setattr(run_leg, "_http_json", fake_http_json)
    monkeypatch.setattr(run_leg, "_run_hook_command", fake_run_hook_command)
    # `_settled_audit_size` re-polls the real audit.log file; stub it to prove
    # `hook_preflight()` calls it AGAIN (with a distinct return value) rather
    # than trusting the value stale-cached from `probe()`.
    monkeypatch.setattr(runner, "_settled_audit_size", lambda **k: 250)

    runner.hook_preflight()

    assert runner.audit_offset == 250
    assert runner.record["audit_offset_after_preflight"] == 250
