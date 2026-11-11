#!/usr/bin/env python3
"""EVAL-LONGITUDINAL-REDISCOVERY -- re-score preserved legs at $0 (DEFECT 9).

Every leg's `transcript.jsonl`, `baselines.json`, and `workspace/` are preserved on
disk as immutable evidence. When the scorer itself changes (DEFECT 9: exec-position
anchoring for BashAction mistake/fact-acquisition signatures + the contradiction
guard), the honest way to see its effect on an already-collected campaign is to
re-run TODAY's `scorer.py` against those preserved artifacts -- no agent, no
daemon, no LLM call, no new dollars spent -- rather than re-run the whole campaign.

Guarantees:
  - Every original `result.json` (and `results.jsonl`) is read-only evidence and is
    NEVER modified. Each leg's re-scored record is written to a NEW sibling file,
    `result.rescored.json`, next to the original. An aggregate `results.rescored.jsonl`
    is written (truncate + rewrite each run, not appended) at the runs-dir root,
    mirroring `results.jsonl`'s own scope (k>=2 legs only -- see `run_leg.py`'s
    `write()`).
  - `score_run()`'s own docstring warns that `VerifyCommand` sub-checks run a
    subprocess with `cwd=workspace` and "may legitimately mutate the workspace" as
    part of verification. Rescoring therefore NEVER points the scorer at a
    preserved `workspace/` directory directly -- it copies that workspace into a
    throwaway `tempfile.TemporaryDirectory()` first, and materializes fresh verify
    scripts into a throwaway verify dir alongside it (exactly as `run_leg.py`'s own
    `score()` does at leg-run time, just replayed against old inputs). Both are
    discarded when the leg finishes rescoring.
  - Trajectory-level fields that depend on the live run itself and cannot be
    replayed from preserved artifacts (`valid`, `invalid_reason`, `non_vacuity`,
    `cost`, and every non-scoring identity field) are copied verbatim from the
    original record. Only `score` (checks + contradictions) and `metrics` are
    recomputed.

Directories named with a `.superseded-` or `.poisoned-` path component (at any
level) are skipped entirely, matching the same exclusion the live campaign itself
already applies to those trees.

Usage:
    rescore.py --runs-dir /path/to/longitudinal-s0
    rescore.py --runs-dir /path/to/longitudinal-s0 --scenario deploy_reverted_by_reconciler
Then: report.py --runs-dir /path/to/longitudinal-s0 --rescored --human
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_THIS_DIR = Path(__file__).resolve().parent
# See report.py / run_leg.py's own comment: benchmarks/injection_utility/ ships a
# same-named scenarios.py/scorer.py, so a bare `import scenarios` races for one
# sys.modules slot. Loading by file path under a fixed, cache-checked key converges
# every caller on ONE module object.
_LONGITUDINAL_SCENARIOS_KEY = "_vectr_eval_longitudinal_scenarios"
_LONGITUDINAL_SCORER_KEY = "_vectr_eval_longitudinal_scorer"
_LONGITUDINAL_REPORT_KEY = "_vectr_eval_longitudinal_report"


def _load_local_module(key: str, filename: str):
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, _THIS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scen = _load_local_module(_LONGITUDINAL_SCENARIOS_KEY, "scenarios.py")
scorer = _load_local_module(_LONGITUDINAL_SCORER_KEY, "scorer.py")
# `report.py` owns the canonical `mistake_rate_post`/`mistake_repetition_rate`
# definition (DESIGN.md 6.5, `mistake_rate_aggregate`) -- reused here rather than
# re-derived, so the before/after comparison uses the exact same per-trajectory
# leg-1-conditioning semantics report.py itself renders.
report = _load_local_module(_LONGITUDINAL_REPORT_KEY, "report.py")


def _is_excluded(path: Path) -> bool:
    # `.superseded-<reason>-<timestamp>` / `.poisoned-<timestamp>` are appended as a
    # SUFFIX onto the trajectory dir's own name (e.g.
    # `deploy_reverted_by_reconciler-none-none-s0.poisoned-20260729T213630Z`), not a
    # separate path component -- substring containment, not prefix, is the right test.
    return any(".superseded-" in part or ".poisoned-" in part for part in path.parts)


def _leg_result_path(traj_dir: Path, leg: Mapping[str, Any]) -> Path | None:
    """Locate a leg's `result.json` from the trajectory directory, not from
    `state.json`'s recorded `result_path`.

    That recorded value is an ABSOLUTE path from the machine that ran the leg, and
    `results/longitudinal/` is a documented COPY of that tree, so on a relocated
    copy it points at a directory that is not this one (or does not exist). Only
    the reconstructed path can keep every rescored sibling inside `runs_dir`; the
    recorded value is consulted last, for layouts the reconstruction does not
    cover (a k=1 leg shared across trajectories lives outside `traj_dir`).
    """
    candidates: list[Path] = []
    k = leg.get("k")
    if k is not None:
        candidates.extend((
            traj_dir / "legs" / str(k) / "artifacts" / "result.json",
            traj_dir / "legs" / str(k) / "result.json",
        ))
    recorded = leg.get("result_path")
    if recorded:
        candidates.append(Path(recorded))
    for candidate in candidates:
        if not _is_excluded(candidate) and candidate.is_file():
            return candidate
    return None


def iter_result_paths(runs_dir: Path):
    """Yield every leg's `result.json` path reachable from a live (non-superseded,
    non-poisoned) trajectory's `state.json`, plus every `_shared/leg1/*/result.json`
    not already reached that way -- deduplicated by resolved path, since a single
    shared leg 1 is referenced by every arm-trajectory at the same scenario+seed.
    """
    seen: set[Path] = set()

    for traj_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if traj_dir.name == "_shared" or _is_excluded(traj_dir):
            continue
        state_path = traj_dir / "state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        for leg in state.get("legs", []):
            p = _leg_result_path(traj_dir, leg)
            if p is None:
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield p

    shared_leg1_dir = runs_dir / "_shared" / "leg1"
    if shared_leg1_dir.is_dir():
        for leg_dir in sorted(p for p in shared_leg1_dir.iterdir() if p.is_dir()):
            if _is_excluded(leg_dir):
                continue
            p = leg_dir / "result.json"
            if not p.is_file():
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield p


def _resolve_artifact_paths(result_path: Path) -> dict[str, Path | None]:
    """`result.json`'s siblings are laid out two ways on disk (DESIGN.md 5.2). The
    shared leg 1 is flat (`transcript.jsonl`/`baselines.json` are direct siblings of
    `result.json`, and its `workspace/` is ALSO a live directory there -- leg 1 never
    gets superseded by a later leg's residue, so the live dir is safe to read
    directly). A k>=2 leg nests `result.json` under `artifacts/`; there is no live
    `workspace/` for it (the trajectory root's own `workspace/` is a single mutable
    directory carried across every leg and reflects only the LAST leg's state, per
    `run_leg.py`'s leg-restore flow -- DEFECT 10's residue concern, not this leg's
    own end state). Each k>=2 leg's own end-of-leg workspace snapshot is instead
    preserved as `end-state.tar` (`_make_tar`, written by `snapshot()` right after
    `score()` runs, so it reflects exactly what was scored) as a SIBLING of
    `artifacts/`. Presence of a live `workspace` subdirectory next to `result.json`
    (not a hardcoded `"artifacts"` name check) is the signal for which layout
    applies; the k>=2 branch instead returns a `workspace_tar` path to extract into a
    scratch directory.
    """
    leg_dir = result_path.parent
    workspace_dir = leg_dir / "workspace"
    if workspace_dir.is_dir():
        return {
            "leg_dir": leg_dir,
            "transcript": leg_dir / "transcript.jsonl",
            "baselines": leg_dir / "baselines.json",
            "workspace": workspace_dir,
            "workspace_tar": None,
        }
    return {
        "leg_dir": leg_dir,
        "transcript": leg_dir / "transcript.jsonl",
        "baselines": leg_dir / "baselines.json",
        "workspace": None,
        "workspace_tar": leg_dir.parent / "end-state.tar",
    }


def _materialize_scratch_workspace(paths: dict[str, Path | None], dest: Path) -> bool:
    """Populate `dest` with the leg's end-of-leg workspace contents, from whichever
    source `_resolve_artifact_paths` found. Returns False (nothing written) when
    neither a live `workspace/` dir nor a `workspace_tar` is available.
    """
    if paths["workspace"] is not None:
        shutil.copytree(paths["workspace"], dest)
        return True
    tar_path = paths["workspace_tar"]
    if tar_path is not None and tar_path.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(dest, filter="data")
        return True
    return False


def rescore_leg(result_path: Path) -> dict[str, Any] | None:
    """Re-run `score_run` / `leg_metrics` / `t3_metrics` / `detect_contradictions`
    against one leg's preserved artifacts, on a scratch copy of its workspace.
    Returns the new rescored record (NOT yet written to disk), or `None` (with a
    stderr warning) when required sibling artifacts are missing -- an older/partial
    leg directory must not abort the whole pass.
    """
    original = json.loads(result_path.read_text())
    paths = _resolve_artifact_paths(result_path)
    if not paths["transcript"].is_file() or not paths["baselines"].is_file():
        print(f"SKIP (missing transcript/baselines): {result_path}", file=sys.stderr)
        return None

    scenario_slug = original["scenario"]
    scenario = scen.SCENARIOS.get(scenario_slug)
    if scenario is None:
        print(f"SKIP (unknown scenario {scenario_slug!r}): {result_path}", file=sys.stderr)
        return None
    k = original["k"]
    leg_spec = scenario.legs[k - 1]
    baselines = json.loads(paths["baselines"].read_text())
    events = scorer.load_transcript(paths["transcript"])
    actions = scorer.build_action_stream(events)

    with tempfile.TemporaryDirectory(prefix="vectr-eval-rescore-") as tmp:
        tmp_path = Path(tmp)
        scratch_workspace = tmp_path / "workspace"
        scratch_verify = tmp_path / "verify"
        if not _materialize_scratch_workspace(paths, scratch_workspace):
            print(f"SKIP (missing workspace/end-state.tar): {result_path}", file=sys.stderr)
            return None
        scen.materialize_verifiers(leg_spec, scratch_verify)

        run_score = scorer.score_run(
            leg_spec, workspace=scratch_workspace, leg_start_baselines=baselines,
            transcript=events, verify_dir=scratch_verify,
        )
        # cost is unaffected by DEFECT 9's fix (a Bash-command-string matching rule);
        # copy it verbatim rather than reparse token-usage events for no reason.
        cost = original.get("cost") or {}
        metrics = scorer.leg_metrics(
            leg_spec, events=events, actions=actions, workspace=scratch_workspace,
            leg_start_baselines=baselines, k=k, origin=scenario.origin,
            session_usd=cost.get("session_usd"),
            billable_tokens_session=cost.get("billable_tokens_session"),
        )
        note_variant_name = original.get("note_variant")
        variant = None
        if note_variant_name and note_variant_name != "none":
            variant = next((v for v in scenario.note_variants if v.variant == note_variant_name), None)
        metrics.update(
            scorer.t3_metrics(
                variant, fact_sentence=scenario.fact_sentence, actions=actions,
                scenario_anchors=scenario.anchor_files(),
            )
        )
        run_score["contradictions"] = scorer.detect_contradictions(
            leg_spec, checks=run_score["checks"], metrics=metrics
        )

    rescored = dict(original)
    rescored["score"] = run_score
    rescored["metrics"] = metrics
    rescored["rescore_meta"] = {
        "rescored_utc": datetime.now(timezone.utc).isoformat(),
        "original_result_path": str(result_path),
    }
    return rescored


def run(runs_dir: Path, scenarios: list[str] | None = None) -> dict[str, Any]:
    """Rescore every reachable leg under `runs_dir`, writing `result.rescored.json`
    next to each original and a fresh `results.rescored.jsonl` (k>=2 legs only,
    truncate + rewrite) at `runs_dir`. Returns a before/after summary dict.

    `scenarios`, if given, restricts only what is INCLUDED in the returned summary
    (and passed on to `report.build_report`) -- rescoring itself, and what gets
    written to `results.rescored.jsonl`, always covers every scenario found under
    `runs_dir`. `results.rescored.jsonl` is truncated and fully rewritten each run
    (matching the module docstring's immutability guarantee only for ORIGINALS);
    a `--scenario`-filtered call would otherwise silently drop every other
    scenario's previously-rescored lines from that shared aggregate file.
    """
    aggregate_lines: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    scenario_filter = set(scenarios) if scenarios else None

    for result_path in iter_result_paths(runs_dir):
        original = json.loads(result_path.read_text())
        rescored = rescore_leg(result_path)
        if rescored is None:
            continue

        out_path = result_path.parent / "result.rescored.json"
        out_path.write_text(json.dumps(rescored, indent=2))

        is_shared_leg1 = "_shared" in result_path.parts and "leg1" in result_path.parts
        if not is_shared_leg1:
            aggregate_lines.append(json.dumps(rescored))

        if scenario_filter is not None and original.get("scenario") not in scenario_filter:
            continue

        before_m = original.get("metrics") or {}
        after_m = rescored.get("metrics") or {}
        summary_rows.append(
            {
                "result_path": str(result_path),
                "scenario": original.get("scenario"),
                "origin": original.get("origin"),
                "trajectory_id": original.get("trajectory_id"),
                "arm": original.get("arm"),
                "k": original.get("k"),
                "mistake_committed_before": before_m.get("mistake_committed"),
                "mistake_committed_after": after_m.get("mistake_committed"),
                "changed": before_m.get("mistake_committed") != after_m.get("mistake_committed"),
                "contradictions_after": rescored.get("score", {}).get("contradictions") or [],
                # DISCOVERED-scenario metrics that must be unaffected by a
                # Bash-command-matching fix; diffed for the red-flag check below.
                "rediscovery_actions_before": before_m.get("rediscovery_actions"),
                "rediscovery_actions_after": after_m.get("rediscovery_actions"),
                "turns_to_fact_before": before_m.get("turns_to_fact"),
                "turns_to_fact_after": after_m.get("turns_to_fact"),
            }
        )

    if aggregate_lines:
        (runs_dir / "results.rescored.jsonl").write_text("\n".join(aggregate_lines) + "\n")

    return _build_summary(runs_dir, summary_rows, scenarios=scenarios)


def _build_summary(
    runs_dir: Path, rows: list[dict[str, Any]], *, scenarios: list[str] | None
) -> dict[str, Any]:
    told_rows = [r for r in rows if r["origin"] == "told"]
    discovered_rows = [r for r in rows if r["origin"] == "discovered"]
    discovered_flipped = [r for r in discovered_rows if r["changed"]]
    contradictions = [r for r in rows if r["contradictions_after"]]

    # `mistake_rate_post`/`mistake_repetition_rate` are per-trajectory (DESIGN.md
    # 6.5: repetition_rate is null unless leg 1 itself committed the mistake) --
    # computed via report.py's own canonical `build_report`, once against the
    # untouched originals and once against the just-written `.rescored` files, so
    # this reuses the exact same tested definition report.py renders.
    before_report = report.build_report(runs_dir, scenarios=scenarios, rescored=False)
    after_report = report.build_report(runs_dir, scenarios=scenarios, rescored=True)
    rate_rows: list[dict[str, Any]] = []
    for slug, sc_after in after_report.items():
        sc_before = before_report.get(slug, {})
        before_trajs = sc_before.get("trajectories", {})
        for traj_id, t_after in sc_after.get("trajectories", {}).items():
            t_before = before_trajs.get(traj_id, {})
            rate_rows.append(
                {
                    "scenario": slug,
                    "origin": sc_after.get("origin"),
                    "trajectory_id": traj_id,
                    "arm": t_after.get("arm"),
                    "mistake_rate_post_before": t_before.get("mistake_rate_post"),
                    "mistake_rate_post_after": t_after.get("mistake_rate_post"),
                    "mistake_repetition_rate_before": t_before.get("mistake_repetition_rate"),
                    "mistake_repetition_rate_after": t_after.get("mistake_repetition_rate"),
                }
            )

    return {
        "legs_rescored": len(rows),
        "told": {"legs": len(told_rows)},
        "discovered": {
            "legs": len(discovered_rows),
            "changed_when_should_not_change": discovered_flipped,
        },
        "new_contradictions": contradictions,
        "rate_rows": rate_rows,
        "rows": rows,
    }


def format_human_readable(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"legs rescored: {summary['legs_rescored']}")
    lines.append("")
    lines.append("TOLD mistake grid (before -> after), per leg:")
    for r in summary["rows"]:
        if r["origin"] != "told":
            continue
        marker = "  <-- FLIPPED" if r["changed"] else ""
        lines.append(
            f"  {r['scenario']:<40} arm={r['arm']:<18} k={r['k']}  "
            f"{r['mistake_committed_before']} -> {r['mistake_committed_after']}{marker}"
        )
    lines.append("")
    lines.append("mistake_rate_post / mistake_repetition_rate (before -> after), per trajectory:")
    for r in summary["rate_rows"]:
        tag = "TOLD" if r["origin"] == "told" else "DISCOVERED"
        lines.append(
            f"  [{tag}] {r['scenario']:<40} arm={r['arm']:<18}  "
            f"rate_post: {r['mistake_rate_post_before']} -> {r['mistake_rate_post_after']}  "
            f"repetition_rate: {r['mistake_repetition_rate_before']} -> {r['mistake_repetition_rate_after']}"
        )
    lines.append("")
    disc = summary["discovered"]
    if disc["changed_when_should_not_change"]:
        lines.append("RED FLAG -- DISCOVERED-scenario per-leg mistake_committed changed (should be impossible):")
        for r in disc["changed_when_should_not_change"]:
            lines.append(f"  {r['scenario']} arm={r['arm']} k={r['k']}: {r}")
    else:
        lines.append(f"DISCOVERED scenarios ({disc['legs']} legs): all mistake_committed values unchanged.")
    disc_rate_drift = [
        r for r in summary["rate_rows"]
        if r["origin"] == "discovered"
        and (
            r["mistake_rate_post_before"] != r["mistake_rate_post_after"]
            or r["mistake_repetition_rate_before"] != r["mistake_repetition_rate_after"]
        )
    ]
    if disc_rate_drift:
        lines.append("RED FLAG -- DISCOVERED-scenario mistake_rate_post/repetition_rate changed:")
        for r in disc_rate_drift:
            lines.append(f"  {r['scenario']} arm={r['arm']}: {r}")
    lines.append("")
    if summary["new_contradictions"]:
        lines.append("New contradiction fields firing:")
        for r in summary["new_contradictions"]:
            for c in r["contradictions_after"]:
                lines.append(f"  {r['scenario']} arm={r['arm']} k={r['k']}: {c['kind']} -- {c['detail']}")
    else:
        lines.append("No contradiction fields fired on the rescored set.")
    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--scenario", action="append", default=None, help="Restrict to one or more scenario slugs; default all.")
    ap.add_argument("--json", action="store_true", help="Print the full machine-readable summary instead of the human table.")
    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()
    summary = run(Path(args.runs_dir), scenarios=args.scenario)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(format_human_readable(summary))


if __name__ == "__main__":
    main()
