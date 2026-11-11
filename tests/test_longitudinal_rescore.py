"""Unit tests for `benchmarks/longitudinal_rediscovery/rescore.py` (DEFECT 9's
$0 re-score entrypoint).

Builds two small on-disk leg fixtures mimicking the real campaign layout
(`/Users/swapnanilsaha/vectr-eval-runs/longitudinal-s0/`, confirmed live): a flat
shared leg 1 (`_shared/leg1/<scenario>-s<seed>/{result.json,transcript.jsonl,
baselines.json,workspace/}`) and a nested k=2 leg (`<traj>/legs/2/artifacts/
{result.json,transcript.jsonl,baselines.json}` + `<traj>/legs/2/end-state.tar` as a
sibling of `artifacts/`, plus `<traj>/state.json` naming both legs' `result_path`s).
Each fixture's ORIGINAL `result.json` carries a stale `mistake_committed: true`
(what the pre-DEFECT-9 scorer produced for a read-only mention); rescoring with
today's anchored scorer must flip it to `false` while leaving the original
byte-identical on disk.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path

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
rescore = _load_local_module("_vectr_eval_longitudinal_rescore_test", "rescore.py")


def _transcript(command: str) -> list[dict]:
    return [
        {"type": "system", "subtype": "init", "mcp_servers": [], "tools": []},
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": command}}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
        {"type": "result", "subtype": "success", "num_turns": 2, "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]


def _write_leg1_fixture(runs_dir: Path, *, scenario_slug: str, seed: int, command: str) -> Path:
    """A flat shared-leg-1 directory with a LIVE `workspace/` (matching the real
    layout's leg 1, which is never superseded by later-leg residue).
    """
    scenario = scen.SCENARIOS[scenario_slug]
    leg = scenario.legs[0]
    leg_dir = runs_dir / "_shared" / "leg1" / f"{scenario_slug}-s{seed}"
    workspace = leg_dir / "workspace"
    baselines = scen.materialize(scenario, workspace)
    (leg_dir / "baselines.json").write_text(json.dumps(baselines))
    (leg_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _transcript(command)) + "\n"
    )
    result = {
        "leg_id": f"stamp-{scenario_slug}-none-none-s{seed}-k1",
        "trajectory_id": f"{scenario_slug}-none-none-s{seed}",
        "scenario": scenario_slug, "arm": "none", "note_variant": "none",
        "seed": seed, "k": 1, "origin": scenario.origin,
        "valid": True, "invalid_reason": "",
        # Stale pre-DEFECT-9 verdict: a read-only mention incorrectly counted.
        "metrics": {"mistake_committed": True, "mistake_source": "action"},
        "score": {"checks": [], "fact_used": False, "contradictions": []},
        "cost": {"session_usd": 0.05},
    }
    (leg_dir / "result.json").write_text(json.dumps(result, indent=2))
    return leg_dir / "result.json"


def _write_leg2_fixture(
    runs_dir: Path, *, scenario_slug: str, seed: int, leg1_result_path: Path, command: str
) -> tuple[Path, Path]:
    """A nested k=2 leg: `<traj>/legs/2/artifacts/*` + a sibling `end-state.tar`
    (`<traj>/legs/2/end-state.tar`), matching `run_leg.py`'s `snapshot()` -- no live
    `workspace/` directory for k>=2, only the tar snapshot.
    """
    scenario = scen.SCENARIOS[scenario_slug]
    leg = scenario.legs[1]
    traj_id = f"{scenario_slug}-none-none-s{seed}"
    traj_dir = runs_dir / f"{scenario_slug}-none-none-s{seed}"
    leg_dir = traj_dir / "legs" / "2"
    artifacts = leg_dir / "artifacts"
    scratch_ws = leg_dir / "_scratch_workspace_for_tar"
    baselines = scen.materialize(scenario, scratch_ws)
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "baselines.json").write_text(json.dumps(baselines))
    (artifacts / "transcript.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _transcript(command)) + "\n"
    )
    result = {
        "leg_id": f"stamp-{scenario_slug}-none-none-s{seed}-k2",
        "trajectory_id": traj_id,
        "scenario": scenario_slug, "arm": "none", "note_variant": "none",
        "seed": seed, "k": 2, "origin": scenario.origin,
        "valid": True, "invalid_reason": "",
        "metrics": {"mistake_committed": True, "mistake_source": "action"},
        "score": {"checks": [], "fact_used": False, "contradictions": []},
        "cost": {"session_usd": 0.07},
    }
    result_path = artifacts / "result.json"
    result_path.write_text(json.dumps(result, indent=2))

    tar_path = leg_dir / "end-state.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(scratch_ws, arcname=".")
    # The tar is the preserved evidence; the scratch dir used only to build it is
    # not part of the real layout (no live workspace/ sibling for a k>=2 leg).
    import shutil as _shutil
    _shutil.rmtree(scratch_ws)

    state = {"legs": [{"k": 1, "result_path": str(leg1_result_path)}, {"k": 2, "result_path": str(result_path)}]}
    (traj_dir / "state.json").write_text(json.dumps(state))
    return result_path, tar_path


# ---------------------------------------------------------------------------
# _is_excluded / iter_result_paths
# ---------------------------------------------------------------------------


def test_is_excluded_matches_superseded_and_poisoned_suffix_anywhere_in_a_component():
    assert rescore._is_excluded(Path("/x/deploy-s0.superseded-defect6-20260730T054015Z/legs/2"))
    assert rescore._is_excluded(Path("/x/deploy-s0.poisoned-20260729T213630Z"))
    assert not rescore._is_excluded(Path("/x/deploy-s0/legs/2"))


def test_iter_result_paths_skips_superseded_and_poisoned_trajectories(tmp_path):
    leg1_path = _write_leg1_fixture(tmp_path, scenario_slug="release_via_ci", seed=0, command="cat RELEASING.md")
    _write_leg2_fixture(tmp_path, scenario_slug="release_via_ci", seed=0, leg1_result_path=leg1_path, command="cat RELEASING.md")

    # A superseded-suffixed sibling trajectory dir with its own state.json must be
    # skipped entirely, exactly as the live campaign's own naming already excludes it.
    excluded_dir = tmp_path / "release_via_ci-none-none-s0.superseded-defect6-20260730T054015Z"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "state.json").write_text(json.dumps({"legs": [{"k": 2, "result_path": str(excluded_dir / "nonexistent.json")}]}))

    found = list(rescore.iter_result_paths(tmp_path))
    assert leg1_path in found
    assert not any(".superseded-" in str(p) for p in found)


def test_iter_result_paths_ignores_stale_absolute_result_path_from_a_relocated_copy(tmp_path):
    """UPG-EVAL-RESCORE-RUNSDIR-ESCAPE: `state.json`'s `result_path` is an ABSOLUTE
    path baked in by `run_plan.py` at the run's ORIGINAL invocation. When `runs_dir`
    passed to `rescore.py` is a relocated COPY of that tree (exactly
    `results/longitudinal/README.md`'s documented "This is a copy of the private run
    directory"), that recorded path points at a directory that may not even exist
    here. `iter_result_paths` must reconstruct the k>=2 path structurally from
    `traj_dir` (already resolved under the CALLER's own `runs_dir`) instead of
    trusting it, so rescoring a relocated copy never escapes outside `runs_dir`.
    """
    leg1_path = _write_leg1_fixture(tmp_path, scenario_slug="release_via_ci", seed=0, command="cat RELEASING.md")
    result_path, _ = _write_leg2_fixture(
        tmp_path, scenario_slug="release_via_ci", seed=0, leg1_result_path=leg1_path, command="cat RELEASING.md",
    )

    # Simulate relocation: state.json still names the ORIGINAL machine's absolute
    # path, which does not exist under this fixture's tmp_path at all.
    traj_dir = result_path.parent.parent.parent.parent
    stale_original_path = "/Users/original-operator/vectr-eval-runs/longitudinal-s0/release_via_ci-none-none-s0/legs/2/artifacts/result.json"
    state = json.loads((traj_dir / "state.json").read_text())
    for leg in state["legs"]:
        if leg["k"] == 2:
            leg["result_path"] = stale_original_path
    (traj_dir / "state.json").write_text(json.dumps(state))

    found = list(rescore.iter_result_paths(tmp_path))
    assert result_path in found  # found via structural reconstruction, not the stale field
    assert not any(str(p) == stale_original_path for p in found)

    summary = rescore.run(tmp_path)
    assert summary["legs_rescored"] == 2
    # Every rescored sibling lands under this call's own runs_dir -- nothing escapes
    # to the stale recorded location (which does not exist and was never created).
    assert (result_path.parent / "result.rescored.json").is_file()
    assert not Path("/Users/original-operator/vectr-eval-runs").exists()


# ---------------------------------------------------------------------------
# rescore_leg / run -- end to end
# ---------------------------------------------------------------------------


def test_rescore_leg_flips_stale_mistake_committed_and_never_touches_the_original(tmp_path):
    leg1_path = _write_leg1_fixture(
        tmp_path, scenario_slug="release_via_ci", seed=0, command="grep -n twine RELEASING.md"
    )
    original_bytes_before = leg1_path.read_bytes()

    rescored = rescore.rescore_leg(leg1_path)
    assert rescored is not None
    assert rescored["metrics"]["mistake_committed"] is False  # was stale-True
    assert rescored["score"]["contradictions"] == []
    # Every non-scoring identity field is preserved verbatim.
    assert rescored["leg_id"] == "stamp-release_via_ci-none-none-s0-k1"
    assert rescored["cost"] == {"session_usd": 0.05}

    assert leg1_path.read_bytes() == original_bytes_before  # original untouched


def test_rescore_leg_extracts_end_state_tar_for_a_k2_leg(tmp_path):
    leg1_path = _write_leg1_fixture(
        tmp_path, scenario_slug="deploy_reverted_by_reconciler", seed=0, command="./deploy.sh staging"
    )
    result_path, tar_path = _write_leg2_fixture(
        tmp_path, scenario_slug="deploy_reverted_by_reconciler", seed=0,
        leg1_result_path=leg1_path, command="cat deploy.sh",
    )
    assert tar_path.is_file()
    original_bytes_before = result_path.read_bytes()

    rescored = rescore.rescore_leg(result_path)
    assert rescored is not None
    assert rescored["metrics"]["mistake_committed"] is False  # cat deploy.sh, not an execution
    assert result_path.read_bytes() == original_bytes_before  # original untouched
    assert tar_path.is_file()  # rescoring never consumes/mutates the preserved tar


def test_run_writes_rescored_siblings_and_results_rescored_jsonl(tmp_path):
    leg1_path = _write_leg1_fixture(
        tmp_path, scenario_slug="release_via_ci", seed=0, command="cat RELEASING.md"
    )
    result_path, _ = _write_leg2_fixture(
        tmp_path, scenario_slug="release_via_ci", seed=0, leg1_result_path=leg1_path,
        command="cat RELEASING.md",
    )

    summary = rescore.run(tmp_path)
    assert summary["legs_rescored"] == 2

    leg1_rescored = leg1_path.parent / "result.rescored.json"
    k2_rescored = result_path.parent / "result.rescored.json"
    assert leg1_rescored.is_file()
    assert k2_rescored.is_file()
    assert json.loads(leg1_rescored.read_text())["metrics"]["mistake_committed"] is False
    assert json.loads(k2_rescored.read_text())["metrics"]["mistake_committed"] is False

    # results.rescored.jsonl carries only the k>=2 leg (leg 1 is shared, matching
    # results.jsonl's own scope per run_leg.py's write()).
    lines = (tmp_path / "results.rescored.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["k"] == 2

    told_flips = [r for r in summary["rows"] if r["changed"]]
    assert len(told_flips) == 2  # both the leg-1 and k=2 rows flip True -> False
