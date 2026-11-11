#!/usr/bin/env python3
"""EVAL-LONGITUDINAL-REDISCOVERY -- run one LEG end to end (DESIGN.md section 11, step 3).

A "leg" is one session in a multi-session trajectory (DESIGN.md sections 2, 5). This
script runs in one of two mutually exclusive modes, matching the two shapes of
DESIGN.md 8's artifact tree exactly:

  --shared-leg1   Leg 1 runs ONCE per (scenario, seed) under arm-A conditions (empty
                  store, `--no-inject`, no MCP, no hooks -- DESIGN.md 5.2) and is
                  reused as the starting snapshot for every arm's leg 1. This IS arm
                  A's own leg 1. Writes self-contained artifacts straight into
                  --out-dir (workspace.tar, manifest.sha256, baselines.json,
                  transcript.jsonl, result.json, leg1_id), with no results.jsonl
                  append -- there is no single trajectory this belongs to yet; a
                  driver (run_plan.py, not yet written) decides how each trajectory
                  incorporates it into its own legs/1/.

  (default)       A normal leg k>=2 (or a standalone k==1, useful for direct
                  debugging/probing without a driver) that reads/writes a single
                  trajectory's `<runs-dir>/<trajectory>/legs/<k>/` and appends its
                  result to `<runs-dir>/results.jsonl`. Per-leg ARTIFACTS
                  (result.json, preflight.json, hook-preflight.json, daemon/proxy logs) always live under
                  that leg's own `legs/<k>/`, but the agent/daemon WORKSPACE is a
                  separate, caller-supplied `--workspace-dir` -- see that flag's help
                  text and `LegRunner._reset_workspace`. run_plan.py always passes
                  the same `<trajectory>/workspace` for every k of one trajectory:
                  vectr's working-memory store scopes notes by workspace path, so a
                  note leg k plants must be registered under the same path leg k+1's
                  fresh daemon queries, even though both legs already share one
                  VECTR_DB_DIR/sqlite file via `--db-dir`. Reusing that path across
                  legs is safe because `prepare()` wipes and re-materializes/restores
                  it fresh on every leg (legs run serially, one at a time).

Isolation and honesty rails mirror `injection_utility/run_harness.py` (one fresh
VECTR_DB_DIR/daemon/proxy process per invocation; scratch ports >= 8899; verify
scripts materialized outside the workspace only after the agent exits) with one
deliberate, DESIGN.md-directed departure: EVERY arm here runs through a
non-injecting `vectr proxy` -- including the hook arms -- because DESIGN.md 4
explicitly requires it ("removing the proxy from the control would confound
injection with proxy transparency"), whereas the trap harness's hook arm has no
proxy on the path at all. Only arm "proxy" (arm C) enables injection.

Non-vacuity gates (DESIGN.md 4.1) are gathered here and handed to
`scorer.leg_non_vacuity` unchanged; the outcome verdict (`scorer.score_run`,
`scorer.leg_metrics`) never sees `arm` at all.

Exit code: 0 whenever the leg ran to completion, INCLUDING an invalid or
task-failing leg -- EXCEPT one specific case: `scorer.leg_non_vacuity`'s
arm-agnostic session-level check (agent session errored / zero output tokens /
nonzero process return code). That is a genuine abort, not a completed-but-invalid
leg -- result.json still records it (valid=false, invalid_reason set) but this
script exits nonzero so a driver (run_plan.py) never chains or caches its end
state. Floors and verdicts otherwise belong to report.py's reader, never to this
script.

Two zero-cost preflight checks run BEFORE the paid `claude -p` session and abort
(nonzero exit, same "fix the scenario, not the score" contract as the exit-code
paragraph above) rather than spend money on a leg that cannot test what it claims
to: `LegRunner.probe()` (every memory arm, daemon-side `/v1/proactive`
reachability) and `LegRunner.hook_preflight()` (arms "hook-sessionstart"/
"hook-full" only, executes the workspace's own configured SessionStart hook
command exactly as `claude` would and asserts both daemon-side
`hook_injection_counts` evidence and planted-content-in-stdout evidence -- see its
own docstring).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_THIS_DIR = Path(__file__).resolve().parent
# `benchmarks/injection_utility/` ships its own same-named `scenarios.py`/`scorer.py`.
# A bare `sys.path.insert` + `import scenarios`/`import scorer` keys off those bare
# names, which both harnesses claim; in a shared interpreter (e.g. this module
# imported alongside the trap harness) whichever wins the race silently poisons the
# other's `sys.modules` entry. Loading by explicit file path under a fixed, private,
# cache-checked key sidesteps the collision. `scorer.py` loads its own `scenarios.py`
# under the identical `_vectr_eval_longitudinal_scenarios` key, so this module and
# `scorer` always converge on ONE `scenarios` module object -- and therefore identical
# check-primitive classes -- regardless of which of the two imports first.
_LONGITUDINAL_SCENARIOS_KEY = "_vectr_eval_longitudinal_scenarios"


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
scorer = _load_local_module("_vectr_eval_longitudinal_scorer", "scorer.py")

DEFAULT_DAEMON_PORT = 8899
DEFAULT_PROXY_PORT = 8900

# Must match leg_result.schema.json's `arm` / `note_variant` enums exactly.
ARMS = ("none", "mcp", "mcp-bare", "proxy", "hook-sessionstart", "hook-full", "hook-userpromptsubmit")
NOTE_VARIANTS = ("none", "plain", "provenance", "verifiable")

# UPG-EVAL-STORE-MATCH (`verify_note_store_matched()` below): the only two arms
# that connect a live vectr MCP server to the agent (mirrors `scorer.py`'s own
# `_EXPECTED_MCP_SERVERS` -- the arms whose expected `system.init.mcp_servers`
# entry is non-empty). Every other arm's agent has no vectr tool at all, so the
# ONLY way a note can ever enter its store is this harness's own single
# `plant_note()` call at k==2 -- for those arms `notes_in_store_at_start` must be
# EXACTLY 1 at every k>=2 leg. These two are excluded from that exact-count
# invariant: an agent with live tool access may legitimately call
# `vectr_remember` during an earlier leg and grow the store past 1 by the time a
# later leg starts, which is real measured behavior, not a harness defect.
_ARMS_WITH_LIVE_TOOL_ACCESS = ("mcp", "mcp-bare")

# Vectr/IDE-owned paths excluded from the drift-verification manifest and from
# leg_start_baselines -- structural and uniform across every scenario/arm/leg
# (never keyed on query content): `.git` for repeated-hashing efficiency across a
# 4-leg trajectory (no scenario check ever targets git internals); the rest are
# config surfaces `vectr start`/`vectr init` may write that no scenario ever
# declares a check against. This is deliberately NOT arm-conditional -- a real
# scenario-content path such as CLAUDE.md is never excluded, even for arm "mcp"
# (the one arm whose CLAUDE.md vectr may rewrite): if a scenario ever checked it,
# suppressing that byte-level signal would be wrong, and drift verification needs
# no such exclusion since a trajectory's own recorded manifest and its own later
# restore always describe the same, consistently-mutated-or-not state.
_MANIFEST_EXCLUDE_DIRS = (".git", ".claude", ".codex", ".vectr", ".cursor", ".vscode")
_MANIFEST_EXCLUDE_FILES = (".mcp.json",)


# ---------------------------------------------------------------------------
# small process / http helpers -- adapted (not imported) from
# injection_utility/run_harness.py's helpers of the same name. Both directories
# define a module literally named `scenarios.py`; run_harness.py's own top-level
# `import scenarios as scen` would rebind the bare `scenarios` module-cache slot
# out from under this file's own `scenarios` import if the two modules were ever
# loaded into the same process (the identical hazard scenarios.py's own
# `_load_trap_harness_scenarios` docstring documents one level down). Duplicating
# ~80 lines of pure stdlib process/http plumbing is the safe side of that
# tradeoff. `_spawn_env_for_agent` is simplified relative to the trap harness's
# version: every arm here keeps a proxy on the path (see module docstring), so
# there is no `base_url=None` case to handle.
# ---------------------------------------------------------------------------


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 30.0):
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {"_http_error": exc.code, "_body": exc.read().decode("utf-8", "replace")[:400]}
    except Exception as exc:
        return {"_error": type(exc).__name__}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"_raw": body[:400]}


def _http_json_with_headers(
    method: str, url: str, payload: dict | None = None, *,
    timeout: float = 30.0, extra_headers: dict[str, str] | None = None,
) -> tuple[dict, dict[str, str]]:
    """Like `_http_json`, but also returns the RESPONSE headers -- needed for
    the MCP streamable-HTTP handshake (DEFECT 13, `_mcp_handshake_probe`
    below): session identity there is carried in the `Mcp-Session-Id`
    response header on `initialize` (`app/routes.py::mcp_jsonrpc`'s own
    docstring), never in a JSON field, so `_http_json`'s body-only return
    can't express it. `urllib`'s `HTTPMessage` header mapping is already
    case-insensitive, matching HTTP's own semantics.
    """
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"} if data else {}
    headers.update(extra_headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            resp_headers = dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        resp_headers = dict(exc.headers.items()) if exc.headers else {}
        return (
            {"_http_error": exc.code, "_body": exc.read().decode("utf-8", "replace")[:400]},
            resp_headers,
        )
    except Exception as exc:
        return {"_error": type(exc).__name__}, {}
    try:
        return json.loads(body), resp_headers
    except json.JSONDecodeError:
        return {"_raw": body[:400]}, resp_headers


def _mcp_handshake_probe(daemon_port: int, *, timeout: float = 30.0) -> dict[str, Any]:
    """Pre-session MCP streamable-HTTP handshake probe (DEFECT 13): `initialize`
    then `tools/list` against the daemon's real `/mcp` JSON-RPC route
    (`app/routes.py::mcp_jsonrpc`), exactly as a compliant MCP client would --
    proves the server `claude -p` is about to connect to over HTTP actually
    serves the vectr tool surface, independent of whatever `system.init`'s
    `mcp_servers` status happens to read at the instant the CLI emits it (an
    http-type server legitimately reads "pending" there while its own async
    connect is still in flight -- see `probe()`'s call site and
    `scorer.leg_non_vacuity`'s docstring).

    Returns `{"initialize": <raw resp>, "tools_list": <raw resp>,
    "session_id": str | None, "ok": bool, "tool_count": int}`. `ok` requires
    BOTH calls to return a JSON-RPC `result` (no `error`, no transport-level
    `_http_error`/`_error`) and `tools_list`'s `result.tools` to contain at
    least one tool name starting with `"vectr_"` (the MCP tool surface's own
    naming convention -- `integrations/mcp_server/_schemas.py`).
    """
    url = f"http://127.0.0.1:{daemon_port}/mcp"
    init_resp, init_headers = _http_json_with_headers(
        "POST", url,
        {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "longitudinal-harness", "version": "1"},
            },
        },
        timeout=timeout,
    )
    session_id = init_headers.get("Mcp-Session-Id") or init_headers.get("mcp-session-id")

    list_headers = {"Mcp-Session-Id": session_id} if session_id else None
    tools_resp, _ = _http_json_with_headers(
        "POST", url,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        timeout=timeout, extra_headers=list_headers,
    )

    def _rpc_ok(resp: dict) -> bool:
        return isinstance(resp, dict) and "result" in resp and "error" not in resp and (
            "_http_error" not in resp and "_error" not in resp
        )

    tools = (tools_resp.get("result") or {}).get("tools") or [] if isinstance(tools_resp, dict) else []
    tool_count = sum(
        1 for t in tools if isinstance(t, dict) and str(t.get("name") or "").startswith("vectr_")
    )
    ok = _rpc_ok(init_resp) and _rpc_ok(tools_resp) and tool_count > 0
    return {
        "initialize": init_resp,
        "tools_list": tools_resp,
        "session_id": session_id,
        "ok": ok,
        "tool_count": tool_count,
    }


def _poll_recall_probe(
    daemon_port: int, query: str, note_id: int | None, *,
    interval_s: float = 2.0, timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Fixes DEFECT 13's recall-probe race: the original single-shot `/v1/recall`
    call fired ~2s after daemon start, before the embedder finished warming, and
    was observed to answer `method=sql`/0 notes even though the SAME query
    against the SAME daemon 26s later (the agent's own in-session recall)
    answered `method=semantic`/1 note (confirmed via VECTR_AUDIT_LOG RECALL
    lines). A cold-embedder miss at probe time is not evidence the memory
    channel is unreachable -- it is evidence the probe ran too early. Poll the
    same query every `interval_s` up to `timeout_s`, stopping as soon as the
    note is found; the elapsed time and the response's own `method` field (when
    present) are recorded for the record, not gated on -- this is a warm-up
    tolerance, not a retrieval-quality check.
    """
    start = time.time()
    last: dict = {}
    while True:
        last = _http_json(
            "POST", f"http://127.0.0.1:{daemon_port}/v1/recall",
            {"query": query}, timeout=60,
        )
        notes_text = last.get("notes") or ""
        returned = (f"[#{note_id}]" in notes_text) if note_id is not None else None
        elapsed = time.time() - start
        if returned or elapsed >= timeout_s:
            return {
                "returned": returned,
                "elapsed_s": round(elapsed, 2),
                "method": last.get("method"),
            }
        time.sleep(interval_s)


def _wait_for(url: str, timeout_s: float, label: str) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        last = _http_json("GET", url, timeout=10.0)
        if "_error" not in last and "_http_error" not in last:
            return last
        time.sleep(1.0)
    raise SystemExit(f"ABORT: {label} did not become ready at {url} within {timeout_s:.0f}s: {last}")


def _port_pids(port: int) -> list[int]:
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(p) for p in out.split() if p.strip().isdigit()]


def _hard_stop_port(port: int) -> list[int]:
    """Last-resort teardown so no scratch listener is ever left running."""
    killed: list[int] = []
    for pid in _port_pids(port):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            time.sleep(1.0)
            if not _port_pids(port):
                break
        killed.append(pid)
    return killed


_STARTED_PORT = re.compile(r"on port (\d+)")


def _parse_started_port(text: str) -> int | None:
    m = _STARTED_PORT.search(text or "")
    return int(m.group(1)) if m else None


def _free_port_at_or_above(start: int, *, avoid: set[int]) -> int:
    for port in range(start, start + 60):
        if port in avoid or _port_pids(port):
            continue
        return port
    raise SystemExit(f"ABORT: no free scratch port at or above {start}")


def _stop_daemon(vectr_bin: str, workspace: Path, port: int, env: dict[str, str]) -> list[int]:
    """`vectr stop` takes `--path`, not `--port` (a `--port` makes argparse exit
    non-zero without stopping anything). The hard-stop fallback closes the gap.
    """
    subprocess.run(
        [vectr_bin, "stop", "--path", str(workspace)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    for _ in range(10):
        if not _port_pids(port):
            return []
        time.sleep(1.0)
    return _hard_stop_port(port)


_SYNTHETIC_GITCONFIG: Path | None = None


def _synthetic_gitconfig() -> Path:
    """Path to a synthetic global git config carrying the synthetic identity.

    `git config user.email` READS resolve through the operator's own
    `~/.gitconfig` unless GIT_CONFIG_GLOBAL says otherwise, so the harness needs
    a real file to point at -- one per harness process, in a private temp dir.
    """
    global _SYNTHETIC_GITCONFIG
    if _SYNTHETIC_GITCONFIG is None:
        directory = Path(tempfile.mkdtemp(prefix="vectr-eval-gitconfig-"))
        config = directory / "gitconfig"
        config.write_text(
            "[user]\n"
            f"\tname = {scen.SYNTHETIC_GIT_USER_NAME}\n"
            f"\temail = {scen.SYNTHETIC_GIT_USER_EMAIL}\n",
            encoding="utf-8",
        )
        _SYNTHETIC_GITCONFIG = config
    return _SYNTHETIC_GITCONFIG


def _spawn_env_for_agent(base_url: str) -> dict[str, str]:
    """Env for the spawned eval agent: every CLAUDE*/ANTHROPIC* var is stripped
    first so the child looks like a fresh user invocation rather than a nested
    session of the harness's own; ANTHROPIC_BASE_URL is set back to route through
    the scratch proxy. OAuth still resolves via the keychain.

    DEFECT 11 PII hygiene (user decision 2026-07-31): every GIT_AUTHOR_*/
    GIT_COMMITTER_* var inherited from the operator's own shell is also
    stripped and re-set to `scenarios.SYNTHETIC_GIT_USER_NAME/EMAIL`. Git's own
    env vars take precedence over BOTH repo-local and global config, so this is
    the layer that covers a git repo the agent inits fresh mid-session (no
    local config pinned yet) -- `scenarios.pin_synthetic_git_identity` is the
    repo-config half of this same defense, applied at materialize/restore time.
    """
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC")
                   or k.startswith("GIT_AUTHOR_") or k.startswith("GIT_COMMITTER_")
                   or k == "GIT_CONFIG_GLOBAL")}
    env["ANTHROPIC_BASE_URL"] = base_url
    env["GIT_AUTHOR_NAME"] = scen.SYNTHETIC_GIT_USER_NAME
    env["GIT_AUTHOR_EMAIL"] = scen.SYNTHETIC_GIT_USER_EMAIL
    env["GIT_COMMITTER_NAME"] = scen.SYNTHETIC_GIT_USER_NAME
    env["GIT_COMMITTER_EMAIL"] = scen.SYNTHETIC_GIT_USER_EMAIL
    # GIT_CONFIG_GLOBAL is stripped (not merely shadowed) and redirected to the
    # synthetic file: authorship env vars do not affect a `git config` READ, so a
    # workspace with no `.git` at all still resolved the operator's real
    # `~/.gitconfig`. NOSYSTEM=1 closes the same leak at the system level.
    env["GIT_CONFIG_GLOBAL"] = str(_synthetic_gitconfig())
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _run_hook_command(
    command: str, *, cwd: str, stdin: str, env: dict[str, str], timeout: float = 60.0,
) -> tuple[int | None, str, str, bool]:
    """Thin, mockable seam around the actual hook-command subprocess spawn --
    `LegRunner.hook_preflight()`'s tests patch THIS (not `subprocess.run` itself),
    mirroring how `_http_json` is the one patchable seam for every daemon call in
    this file. `command` is the exact `command` string `.claude/settings.json`'s
    SessionStart hook group carries (main.py writes a bare shell command, e.g.
    "vectr hook session-start"), so `shell=True` -- the same way Claude Code's own
    hook runner would exec it, not our own argv-splitting guess. Returns
    (returncode, stdout, stderr, timed_out); returncode is None only on timeout.
    """
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, input=stdin,
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or "") + f"\n[hook_preflight] timed out after {timeout:.0f}s"
        return None, (exc.stdout or ""), stderr, True


# `vectr hook <event>`'s own instance-resolution registry (agent/instance_registry.py):
# a global, machine-wide `{workspace_hash(dir): {port, ...}}` map, walked from the
# hook's cwd up through parents until a hit. Read directly here (not imported --
# see the module docstring's rationale for duplicating run_harness.py helpers rather
# than importing product internals: this file must give identical answers regardless
# of which worktree's Python happens to run it, and an editable install's finder maps
# `import agent.*` to ONE fixed checkout path irrespective of caller worktree) purely
# as a DIAGNOSTIC on preflight failure -- never a pass/fail gate itself. The gate is
# the daemon-side hook_injection_counts delta below, which already catches a wrong
# resolution structurally (the count never moves on THIS leg's own daemon); this just
# names the culprit port when it does, instead of leaving it a mystery in the artifact.
_VECTR_INSTANCE_REGISTRY_PATH = Path.home() / ".vectr" / "instances.json"


def _workspace_hash(path: str) -> str:
    """Mirrors `agent.instance_registry.workspace_hash` (sha256(path)[:12]) --
    reimplemented locally rather than imported; see `_VECTR_INSTANCE_REGISTRY_PATH`.
    """
    return hashlib.sha256(path.encode()).hexdigest()[:12]


def _registry_port_for(cwd: Path) -> int | None:
    """Best-effort: which port would `vectr hook <event>` resolve for `cwd` per the
    global instance registry, walking `cwd` then its parents (same order as
    `agent.instance_registry`'s own resolution) until a hit. None on any read/parse
    failure or no match -- diagnostic only, see module comment above.
    """
    try:
        data = json.loads(_VECTR_INSTANCE_REGISTRY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    here = cwd.resolve()
    for d in (here, *here.parents):
        entry = data.get(_workspace_hash(str(d)))
        if entry is not None:
            return entry.get("port")
    return None


# ---------------------------------------------------------------------------
# tree hashing, manifests, tar snapshots
# ---------------------------------------------------------------------------


def _sha256_tree(root: Path, *, extra_ignore: Sequence[str] = ()) -> dict[str, str]:
    """{relpath: sha256} for every regular file under root, skipping the fixed
    vectr/IDE-owned exclusion set plus any scenario-declared `ignore_paths`. Used
    both as `leg_start_baselines` fed to `scorer.score_run`/`leg_metrics` (a
    superset of `scenarios.materialize`'s narrower dict -- also covers files an
    earlier leg's agent created) and as the drift-verification manifest input.
    """
    ignore_files = set(_MANIFEST_EXCLUDE_FILES) | set(extra_ignore)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _MANIFEST_EXCLUDE_DIRS for part in rel.parts[:-1]):
            continue
        if str(rel) in ignore_files or rel.name in ignore_files:
            continue
        out[str(rel)] = scen.sha256_file(path)
    return out


def _write_manifest(baselines: Mapping[str, str], manifest_path: Path) -> str:
    """Write a sorted `{relpath}  {sha256}` manifest file and return the sha256 OF
    THAT FILE'S BYTES -- a single comparable hash matching
    trajectory_state.schema.json's `end_state_manifest_sha256: string`, while the
    underlying manifest file stays fully inspectable/diffable on disk.
    """
    lines = [f"{rel}  {digest}\n" for rel, digest in sorted(baselines.items())]
    manifest_path.write_text("".join(lines), encoding="utf-8")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _make_tar(workspace: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w") as tf:
        tf.add(workspace, arcname=".")


def _extract_tar(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dest, filter="data")


def _scrub_ide_config_marker(workspace: Path) -> None:
    """Delete `.vectr/ide_config` (main.py's `_IDE_CONFIG_MARKER_REL`) if a
    restored snapshot carries one forward.

    That marker PERSISTS a workspace's IDE-config choice across separate
    `vectr start` calls for a real interactive user
    (`_persist_ide_config_disabled` / `_ide_config_disabled`, main.py) -- exactly
    the wrong semantics here, where every leg re-passes its own arm-appropriate
    `--no-ide-config` choice on every `vectr start` call. Left in place, leg 1's
    `--no-ide-config` (arm-A conditions, DESIGN.md 5.2) would silently suppress
    arm "mcp"'s CLAUDE.md/.mcp.json auto-write on every one of that trajectory's
    legs, since every trajectory's leg 1 starts from either the shared leg-1 tar
    or (for k>1) its own prior leg's restored snapshot.
    """
    (workspace / ".vectr" / "ide_config").unlink(missing_ok=True)


def _collapse_ws(text: str) -> str:
    """Whitespace-normalize for delivery-containment checks (DEFECT 8): every
    channel that renders a note into injected/hook context collapses it to one
    line first (`agent/proactive/matcher.py::_one_line`, `" ".join(text.split())`
    -- deliberate product behavior, confirmed by the anti-memory lane; reimplemented
    locally rather than imported, for the same worktree-path-pinning reason as
    `_workspace_hash` above). A `NoteVariant.content`/trail string is authored text
    and may legitimately contain internal newlines (e.g. a multi-line provenance
    trail); a literal `x in y` substring check against the ORIGINAL (un-collapsed)
    string is then structurally false even when delivery succeeded. Every site in
    this file (and `scorer.py`) that asserts "was this content delivered" via `in`
    must normalize BOTH sides with this helper first, not just the delivered side --
    otherwise the check depends on the accident that today's variant strings happen
    to be single-line.
    """
    return " ".join(text.split())


def _content_delivered_in_json_text(content: str, raw_text: str) -> bool:
    """`_collapse_ws`-based containment check for a haystack that is (or
    embeds) JSON-serialized text -- `hook_preflight()`'s raw subprocess
    stdout (`agent/hook_cli.py::_emit_hook_context` does `print(json.dumps({
    "hookSpecificOutput": {..., "additionalContext": text}}))`) and
    `scorer.leg_non_vacuity`'s raw stream-json transcript file both embed
    the delivered note text as the VALUE of a JSON string field, so an
    internal newline in the source content is JSON-escaped there to the two
    literal characters `\\n` -- whitespace-collapsing the raw haystack alone
    (`_collapse_ws`) cannot match that back to the source content's real
    newline, since `\\` and `n` are not whitespace. This checks the content
    against the haystack in BOTH its literal form (a haystack that embeds it
    unescaped, e.g. non-JSON transcript prose) and its JSON-string-escaped
    form (`json.dumps(content)[1:-1]`, i.e. what `content` renders as once
    placed inside a JSON string), collapsing whitespace on both sides of
    each comparison exactly like `_collapse_ws` does elsewhere in this file.
    """
    if not content:
        return False
    collapsed_haystack = _collapse_ws(raw_text)
    if _collapse_ws(content) in collapsed_haystack:
        return True
    escaped = json.dumps(content)[1:-1]
    return _collapse_ws(escaped) in collapsed_haystack


def _proactive_probe(
    daemon_port: int, leg: "scen.LegSpec", workspace: Path, session_prefix: str,
) -> tuple[dict, dict]:
    """Two daemon-side `/v1/proactive` reachability probes, mirroring
    `injection_utility/run_harness.py`'s `Cell.probe`: turn 1 is text-only (no
    structural anchors exist before any tool call -- `assemble_window` reads paths
    only from tool-input keys, never free text); turn 2 adds the leg's declared
    `probe_files` as file-path anchors. Session ids are synthetic and never
    collide with a real agent session, and this never touches the proxy (a
    daemon-only call), so it cannot consume the proxy's own cooldown ledger.
    """
    base = f"http://127.0.0.1:{daemon_port}/v1/proactive"
    turn1 = _http_json("POST", base, {
        "text": leg.prompt, "file_paths": [], "symbols": [],
        "session_id": f"{session_prefix}-turn1", "channel": "proxy",
    }, timeout=60)
    turn2 = _http_json("POST", base, {
        "text": leg.prompt,
        "file_paths": [str(workspace / p) for p in leg.probe_files],
        "symbols": [],
        "session_id": f"{session_prefix}-turn2", "channel": "proxy",
    }, timeout=60)
    return turn1, turn2


def _transport_ok(resp: dict) -> bool:
    """True when `_http_json` got a real HTTP response body, as opposed to a
    transport-level failure (`_error`: the request never got a response at
    all, e.g. connection refused; `_http_error`: an HTTP error status;
    `_raw`: the body wasn't valid JSON). Used to tell "the daemon answered
    but said no" apart from "the daemon (or embedder) never answered" --
    UPG-EVAL-PLANT-DISPLACEMENT's infra-unreachable ABORT criterion below
    depends on this distinction, not just a truthy/falsy response."""
    return isinstance(resp, dict) and "_error" not in resp and "_http_error" not in resp and "_raw" not in resp


def _note_by_id_probe(daemon_port: int, note_id: int, expected_content: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Direct daemon-side by-id existence/integrity probe for a planted note
    (UPG-EVAL-PLANT-DISPLACEMENT) -- the PRIMARY reachability test for every
    non-hook memory arm, replacing a channel-ranking-based reachability gate.
    `/v1/recall` with `note_id` set (`app/service.py`'s single-note-expand
    branch, UPG-RECALL-HIERARCHY) reads the note directly by id, independent
    of any channel's matching/ranking/budget -- existence and content
    integrity, not deliverability through one particular channel at one
    particular moment. `app/service.py`'s own not-found sentinel is the
    literal string `f"Note #{note_id} not found."`; a found note's block
    starts with `[{note_id}] [...`. Comparing against the DECODED REST field
    (`_store.py::_format_full_block`'s rendered text, already JSON-decoded
    by `_http_json`) needs only `_collapse_ws`, not the JSON-escape-aware
    `_content_delivered_in_json_text` (that helper is for RAW haystacks that
    themselves embed a JSON-escaped string field -- see its own docstring).

    Returns `{"transport_ok": bool, "exists": bool, "content_matches": bool,
    "raw_chars": int}`. A transport failure (daemon unreachable) yields
    `exists=False`/`content_matches=False` rather than raising -- the caller
    tells "genuinely absent" apart from "infra broken" by ALSO consulting
    `transport_ok` (and the corroborating proactive probe's own transport
    status), per this module's `probe()` docstring.
    """
    out = _http_json(
        "POST", f"http://127.0.0.1:{daemon_port}/v1/recall", {"note_id": note_id}, timeout=timeout,
    )
    transport_ok = _transport_ok(out)
    notes_text = (out.get("notes") or "") if transport_ok else ""
    not_found = notes_text.strip() == f"Note #{note_id} not found."
    exists = transport_ok and notes_text.strip() != "" and not not_found
    content_matches = bool(
        exists and expected_content and _collapse_ws(expected_content) in _collapse_ws(notes_text)
    )
    return {
        "transport_ok": transport_ok,
        "exists": exists,
        "content_matches": content_matches,
        "raw_chars": len(notes_text),
    }


# Parses the FIXED candidate-line format `agent/proactive/matcher.py`'s
# `_structural_note_candidate`/`_semantic_note_candidate` render into a
# `/v1/proactive` response's own `context` field: `note #{id} ({kind},
# {provenance}[, anchored to|mentions {anchor}]): {summary}`. Reimplemented
# locally against the product's own deterministic OUTPUT format (never
# imported, never query-content-conditional) -- the same established
# convention as this file's `_STARTED_PORT`/`_content_delivered_in_json_text`.
_CANDIDATE_LINE_RE = re.compile(
    r"^note #(?P<id>\d+) \((?P<kind>[^,]+), (?P<provenance>[^,)]+)"
    r"(?:, (?P<relation>anchored to|mentions) (?P<anchor_label>.+?))?\):"
)

# Parses the FIXED full-block header format `agent/working_context_store/
# _store.py`'s `_format_full_block` renders for `/v1/recall(note_id=...)`:
# `[{id}] [{PRIORITY}]{ [KIND]}? [{provenance}] ...`. `kind` is only present
# when it differs from the default ("finding").
_NOTE_HEADER_RE = re.compile(
    r"^\[(?P<id>\d+)\] \[(?P<priority>[A-Z]+)\](?: \[(?P<kind>[A-Z]+)\])? \[(?P<provenance>[a-zA-Z]+)\]"
)


def _note_header_fields(daemon_port: int, note_id: int, *, timeout: float = 15.0) -> dict[str, Any]:
    """Best-effort priority for one DISPLACING note id (`_proactive_rank_probe`
    below), parsed from `/v1/recall(note_id=...)`'s own fixed header format
    (`_NOTE_HEADER_RE`). Diagnostic metadata only, never a gate: a transport
    hiccup or parse miss on one displacing note's lookup yields
    `priority=None` rather than raising -- it must never abort a leg that
    the primary by-id check already confirmed reachable.
    """
    out = _http_json(
        "POST", f"http://127.0.0.1:{daemon_port}/v1/recall", {"note_id": note_id}, timeout=timeout,
    )
    notes_text = (out.get("notes") or "") if _transport_ok(out) else ""
    m = _NOTE_HEADER_RE.match(notes_text.strip())
    return {"priority": m.group("priority").lower() if m else None}


def _proactive_rank_probe(
    daemon_port: int, leg: "scen.LegSpec", workspace: Path, session_prefix: str,
    planted_anchor: str, cap: int,
) -> dict[str, Any]:
    """Finds `planted_anchor`'s rank in the proactive channel when it is
    absent from the channel's default-budget response (UPG-EVAL-PLANT-
    DISPLACEMENT) -- called only after `_note_by_id_probe` has already
    confirmed the note exists and is intact, so this is diagnostic ranking,
    never a reachability gate on its own.

    `/v1/proactive` has no `items`/`max_items` request override (its budget,
    `proactive.max_items_per_event` in agent/config.yaml, is a static
    per-workspace config value); this instead PAGINATES by repeating the
    identical request against ONE synthetic `session_id`.
    `ProactiveGate.select()` charges its cooldown ledger at retrieval by
    default (`agent/proactive/gate.py`), so a candidate returned (and
    charged) on round 1 is cooldown-suppressed on round 2 against the SAME
    session_id, surfacing the next tier of eligible candidates -- an
    existing product mechanism, read here purely as a harness-side
    diagnostic technique, never a new request parameter and never touching
    the real agent's own (differently-named) session_id, so this cannot
    affect the paid leg's own delivery.

    Each accumulated candidate's kind and anchored-ness are parsed from the
    SAME response's `context` field via `_CANDIDATE_LINE_RE` -- never a new
    call, never query content. Priority for each note ranked above the
    plant is filled in via `_note_header_fields` (bounded by `cap`, so at
    most `cap` extra zero-LLM-cost daemon calls). Stops as soon as the
    plant is found, `cap` candidates have been accumulated, or a round
    returns nothing further (the channel is exhausted for this session).

    Returns `{"rank": int | None, "rounds": int, "exhausted": bool,
    "accumulated_anchor_ids": [...], "displaced_by": [{"note_id", "kind",
    "priority", "anchored"}, ...]}` -- `rank` is None when the plant never
    surfaced within `cap` (still not a gate failure: the primary by-id
    check already established the note is reachable in principle).
    """
    base = f"http://127.0.0.1:{daemon_port}/v1/proactive"
    session_id = f"{session_prefix}-rankprobe"
    accumulated: list[dict[str, Any]] = []
    rounds = 0
    exhausted = False
    while len(accumulated) < cap:
        resp = _http_json("POST", base, {
            "text": leg.prompt,
            "file_paths": [str(workspace / p) for p in leg.probe_files],
            "symbols": [],
            "session_id": session_id, "channel": "proxy",
        }, timeout=60)
        rounds += 1
        anchor_ids = resp.get("anchor_ids") or []
        if not anchor_ids:
            exhausted = True
            break
        by_id: dict[str, dict[str, Any]] = {}
        for line in (resp.get("context") or "").splitlines():
            m = _CANDIDATE_LINE_RE.match(line.strip())
            if m:
                by_id[f"note:{m.group('id')}"] = {
                    "kind": m.group("kind"),
                    "anchored": m.group("relation") == "anchored to",
                }
        for aid in anchor_ids:
            entry = {"anchor_id": aid, **by_id.get(aid, {"kind": None, "anchored": None})}
            accumulated.append(entry)
        if planted_anchor in anchor_ids:
            break
        if rounds > cap:  # safety valve against a misbehaving daemon looping forever
            break

    rank = next(
        (i + 1 for i, e in enumerate(accumulated) if e["anchor_id"] == planted_anchor), None
    )
    above = accumulated if rank is None else accumulated[: rank - 1]
    displaced_by = []
    for e in above:
        try:
            nid = int(e["anchor_id"].split(":", 1)[1])
        except (IndexError, ValueError):
            nid = None
        priority = _note_header_fields(daemon_port, nid)["priority"] if nid is not None else None
        displaced_by.append({
            "note_id": nid, "kind": e.get("kind"), "priority": priority, "anchored": e.get("anchored"),
        })

    return {
        "rank": rank,
        "rounds": rounds,
        "exhausted": exhausted,
        "accumulated_anchor_ids": [e["anchor_id"] for e in accumulated],
        "displaced_by": displaced_by,
    }


def _session_start_probe(daemon_port: int, session_id: str) -> str:
    """Daemon-side SessionStart-channel reachability check (DEFECT 7): a direct
    `/v1/recall` POST carrying the exact `{"boot": True, "hook_event":
    "SessionStart"}` payload `agent/hook_cli.py::run_hook`'s own "session-start"
    branch sends on a real (non-compaction) SessionStart -- so this reaches the
    identical `VectrService._recall_impl` boot branch a real hook invocation
    reaches. Channel-true and zero-LLM-cost, and unlike `LegRunner.
    hook_preflight()`'s subprocess-based check, this needs neither
    `write_hooks()` (no `.claude/settings.json` yet -- `probe()` runs first) nor
    a real hook command round trip; it is a pure daemon call, mirroring
    `_proactive_probe`'s own daemon-only shape.

    Returns the raw notes text exactly as the daemon renders it into the real
    hook's `additionalContext` envelope -- already JSON-decoded by `_http_json`
    (this is a `/v1/recall` REST response field, not the hook subprocess's own
    JSON-encoded stdout), so a caller compares it with `_collapse_ws` on both
    sides (DEFECT 8), not `_content_delivered_in_json_text` (that one is for
    `hook_preflight()`'s raw stdout / `scorer.py`'s raw transcript file, where
    the content is still JSON-escaped in the haystack).
    """
    out = _http_json(
        "POST", f"http://127.0.0.1:{daemon_port}/v1/recall",
        {"boot": True, "hook_event": "SessionStart", "session_id": session_id},
        timeout=60,
    )
    return out.get("notes") or ""


def _user_prompt_submit_probe(daemon_port: int, session_id: str, query: str) -> str:
    """Daemon-side UserPromptSubmit-channel reachability check (DEFECT 7,
    arm "hook-userpromptsubmit" -- the same pattern `_session_start_probe`
    above uses for "hook-sessionstart"/"hook-full"): a direct `/v1/recall`
    POST carrying the exact `{"query": ..., "hook_event": "UserPromptSubmit",
    "events": ["prompt-submit"]}` payload `agent/hook_cli.py::run_hook`'s own
    "user-prompt-submit" branch sends on a real prompt submission -- so this
    reaches the identical `VectrService._recall_impl` generic-query branch (a
    trigger-fired pass plus an ordinary ranked `recall()` pass, neither gated
    by note kind) a real hook invocation reaches. Channel-true and
    zero-LLM-cost, and like `_session_start_probe` it is a pure daemon call
    needing neither `write_hooks()` nor a real hook subprocess round trip.

    Returns the raw notes text exactly as the daemon renders it -- a caller
    compares it with `_collapse_ws` on both sides (DEFECT 8), same as
    `_session_start_probe`'s own contract. This is a pre-spend daemon HTTP
    response, never a `claude -p` transcript, so it is unaffected by
    UPG-IU-HOOK-NONVACUITY-CANARY (that bug is about content never rendering
    into a *stream-json transcript*, not about this direct `/v1/recall` call).
    """
    out = _http_json(
        "POST", f"http://127.0.0.1:{daemon_port}/v1/recall",
        {
            "query": query, "hook_event": "UserPromptSubmit",
            "events": ["prompt-submit"], "session_id": session_id,
        },
        timeout=60,
    )
    return out.get("notes") or ""


def _enforce_hook_attestation(arm: str, attestation_path: str | None) -> dict | None:
    """DESIGN.md 4: arm `hook-full` (D2) is SKIPPED, never run/scored, without a
    fresh canary attestation -- D2's UserPromptSubmit/PreToolUse additionalContext
    never renders in a stream-json transcript (UPG-IU-HOOK-NONVACUITY-CANARY), so
    delivery must be independently attested rather than read from the transcript.
    """
    if arm != "hook-full":
        return None
    if not attestation_path:
        raise SystemExit(
            "ABORT: --hook-attestation is required for --arm hook-full "
            "(DESIGN.md 4); refusing to run an unattested D2 leg."
        )
    path = Path(attestation_path)
    if not path.is_file():
        raise SystemExit(f"ABORT: --hook-attestation file not found: {path}")
    try:
        att = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ABORT: --hook-attestation is not valid JSON: {exc}")
    if att.get("verified") is not True:
        raise SystemExit(
            f"ABORT: --hook-attestation {path} does not have verified:true; "
            f"arm hook-full is SKIPPED, never run, without a fresh canary attestation."
        )
    return att


# ---------------------------------------------------------------------------
# LegRunner
# ---------------------------------------------------------------------------


class LegRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.scenario = scen.get(args.scenario)
        self.seed = args.seed
        self.shared_leg1 = bool(args.shared_leg1)
        self.k = 1 if self.shared_leg1 else args.k
        self.arm = "none" if self.shared_leg1 else args.arm
        self.note_variant = "none" if self.shared_leg1 else args.note_variant
        if not (1 <= self.k <= len(self.scenario.legs)):
            raise SystemExit(
                f"ABORT: scenario {self.scenario.slug!r} has {len(self.scenario.legs)} "
                f"legs; k={self.k} out of range"
            )
        self.leg_spec = self.scenario.legs[self.k - 1]
        # leg_result.schema.json documents trajectory_id as
        # "<scenario>-<arm>-<variant>-s<seed>" -- shared-leg1 has already forced
        # arm="none"/note_variant="none" above, so this is the SAME formula for
        # both modes: shared-leg1 IS conceptually k=1 of the "none" trajectory,
        # which is exactly what every other arm's own leg 1 restores from.
        self.trajectory_id = (
            args.trajectory_id if not self.shared_leg1
            else f"{self.scenario.slug}-none-none-s{self.seed}"
        )

        self.root = Path(args.out_dir if self.shared_leg1 else args.leg_dir).resolve()
        # `self.workspace` is the agent/daemon workspace -- the directory `vectr
        # start`/the spawned `claude -p` cwd/the drift manifest all operate on. It is
        # DELIBERATELY NOT always `self.root / "workspace"`: vectr's working-memory
        # store scopes notes by workspace PATH (not by db-dir alone), so a normal
        # (non-shared-leg1) leg k+1 must register the SAME workspace path leg k used,
        # or a note that leg k planted is invisible to leg k+1's fresh daemon even
        # though both legs share one VECTR_DB_DIR/sqlite file. `--workspace-dir` lets
        # a driver (run_plan.py) pass one trajectory-stable path
        # (`<trajectory-dir>/workspace`) across every leg of that trajectory; a
        # caller that omits it (standalone k==1 debugging, T0's single-shot probe
        # cells, --shared-leg1) keeps the prior per-`self.root` default, which is
        # correct there because those cases never chain a planted note across a
        # workspace-path change.
        self.workspace = (
            Path(args.workspace_dir).resolve()
            if (not self.shared_leg1 and getattr(args, "workspace_dir", None))
            else self.root / "workspace"
        )
        self.artifacts = self.root if self.shared_leg1 else (self.root / "artifacts")
        self.verify_dir = self.root / "verify"
        self.audit_log = self.artifacts / "audit.log"
        self.db_dir = (
            (self.root / "_scratch_db") if self.shared_leg1 else Path(args.db_dir).resolve()
        )

        self.daemon_port = args.daemon_port
        self.proxy_port = args.proxy_port
        self.vectr = args.vectr_bin
        self.proxy_proc: subprocess.Popen | None = None
        self._proxy_stderr = None

        self.note_id: int | None = None if self.shared_leg1 else args.planted_note_id
        self.planted_anchor: str | None = None if self.shared_leg1 else args.planted_anchor
        self.audit_offset = 0
        self.leg_start_baselines: dict[str, str] = {}
        self.notes_count_at_start: int | None = None
        self.restored_manifest_ok: bool | None = None
        self.trail_text_delivered: bool | None = None
        self.recall_probe_returned_note: bool | None = None
        self.mcp_handshake_ok: bool | None = None

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # leg_result.schema.json documents leg_id as
        # "<stamp>-<scenario>-<arm>-<variant>-s<seed>-k<k>".
        leg_id = f"{stamp}-{self.scenario.slug}-{self.arm}-{self.note_variant}-s{self.seed}-k{self.k}"
        self.record: dict[str, Any] = {
            "leg_id": leg_id,
            "trajectory_id": self.trajectory_id,
            "scenario": self.scenario.slug,
            "arm": self.arm,
            "note_variant": self.note_variant,
            "seed": self.seed,
            "k": self.k,
            "model": args.model,
            "started_utc": stamp,
            "origin": self.scenario.origin,
            # Nullable/optional per schema; pre-populated so every result.json has
            # a stable key set for report.py's reader regardless of which methods
            # this particular leg ends up calling.
            "leg1_id": None,
            "restored_manifest_ok": None,
            "planted_note_id": self.note_id,
            "planted_anchor": self.planted_anchor,
            "notes_in_store_at_start": None,
            "mcp_handshake_ok": None,
            "mcp_handshake_tools": None,
            "recall_probe_method": None,
            "recall_probe_elapsed_s": None,
            "recall_probe_returned_note": None,
            # UPG-EVAL-PLANT-DISPLACEMENT: displacement diagnostics, non-hook
            # memory arms only (None elsewhere -- see `probe()`).
            "planted_rank": None,
            "displaced_by": None,
            "delivered_at_default": None,
            "channel_delivery": None,
        }

    # -- small path helper -------------------------------------------------

    def _out(self, name: str) -> Path:
        return self.artifacts / name

    def _daemon_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["VECTR_DB_DIR"] = str(self.db_dir)
        env["VECTR_AUDIT_LOG"] = str(self.audit_log)
        env["VECTR_PORT"] = str(self.daemon_port)
        env["VECTR_WORKSPACE"] = str(self.workspace)
        return env

    def _extra_ignore_paths(self) -> tuple[str, ...]:
        return tuple(self.scenario.ignore_paths)

    def _find_note_variant(self) -> "scen.NoteVariant":
        for v in self.scenario.note_variants:
            if v.variant == self.note_variant:
                return v
        raise SystemExit(
            f"ABORT: scenario {self.scenario.slug!r} has no note_variant "
            f"{self.note_variant!r} (has: {[v.variant for v in self.scenario.note_variants]})"
        )

    # -- setup --------------------------------------------------------------

    def _reset_workspace(self) -> None:
        """Wipe and recreate `self.workspace` before every leg's materialize/restore
        step. Required now that `--workspace-dir` can make this path TRAJECTORY-
        stable (reused across k=1..N, not a fresh never-before-seen directory per
        leg): without a wipe, leg k+1's materialize/`_extract_tar` would layer onto
        whatever leg k's agent left behind, so a file leg k's agent deleted (and
        therefore absent from its restore-tar) would wrongly persist into leg k+1.
        A no-op the first time any given workspace path is used (nothing to wipe).
        """
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def prepare(self) -> None:
        if self.root.exists() and any(self.root.iterdir()) and not self.args.probe_only:
            raise SystemExit(
                f"ABORT: output dir already has content: {self.root} -- resumability "
                f"is the driver's job; it must not re-invoke run_leg.py for a leg "
                f"whose result already exists (DESIGN.md 8)"
            )
        for d in (self.artifacts, self.verify_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._reset_workspace()

        if self.k == 1 or (self.args.probe_only and not self.args.restore_tar):
            self.leg_start_baselines = scen.materialize(self.scenario, self.workspace)
        else:
            self._restore_and_verify()

        if not self.shared_leg1:
            self.db_dir.mkdir(parents=True, exist_ok=True)

        _scrub_ide_config_marker(self.workspace)
        # DEFECT 11 PII hygiene: pin the synthetic git identity uniformly for
        # every k, not just the k==1 materialize() branch above (which already
        # pins it internally) -- a k>=2 leg's `_restore_and_verify` extracts a
        # tar rather than calling `materialize()`, so this is the only place
        # that covers both branches with one call. Idempotent (re-setting the
        # same config values); a no-op for a non-git scenario.
        scen.pin_synthetic_git_identity(self._agent_cwd())
        self._out("baselines.json").write_text(json.dumps(self.leg_start_baselines, indent=2))
        self._out("scenario.json").write_text(json.dumps({
            "slug": self.scenario.slug,
            "title": self.scenario.title,
            "origin": self.scenario.origin,
            "corroborable": self.scenario.corroborable,
            "k": self.k,
            "prompt": self.leg_spec.prompt,
            "primary_check": self.leg_spec.primary_check,
            "checks": [repr(c) for c in self.leg_spec.checks],
            "is_forcing_leg": self.leg_spec.is_forcing_leg,
            "critical_residue_paths": list(self.scenario.critical_residue_paths),
        }, indent=2))

    def _restore_and_verify(self) -> None:
        tar_path = Path(self.args.restore_tar).resolve()
        if not tar_path.is_file():
            raise SystemExit(f"ABORT: --restore-tar not found: {tar_path}")
        _extract_tar(tar_path, self.workspace)
        _scrub_ide_config_marker(self.workspace)
        baselines = _sha256_tree(self.workspace, extra_ignore=self._extra_ignore_paths())
        manifest_hash = _write_manifest(baselines, self._out("restored-manifest.sha256"))
        expected = self.args.restore_manifest_sha256
        ok = manifest_hash == expected
        self.restored_manifest_ok = ok
        self.record["restored_manifest_ok"] = ok
        if not ok and not self.args.allow_manifest_mismatch:
            raise SystemExit(
                f"ABORT: restored snapshot manifest mismatch for {tar_path} "
                f"(got {manifest_hash}, expected {expected}) -- refusing to continue "
                f"on drifted state; pass --allow-manifest-mismatch to override for "
                f"debugging"
            )
        self.leg_start_baselines = baselines
        self._apply_critical_residue_reset()

    def _apply_critical_residue_reset(self) -> None:
        """DEFECT 10 (direction 1, user decision 2026-07-30; DESIGN.md 6.5): restore
        this scenario's declared `critical_residue_paths` to their scenario-seed
        content before this leg's agent starts.

        Runs only from `_restore_and_verify` (i.e. only at k>=2 -- k==1 already
        starts from the seed via `scen.materialize`), and only AFTER the manifest
        integrity check above, so the DEFECT-9-adjacent tar-fidelity verification
        still covers the raw, un-reset restore; this reset is a deliberate,
        scenario-authored transformation layered on top of an already-verified-
        intact restore, never a correction to a corrupted one.

        Every path NOT declared here keeps its natural cross-leg residue untouched
        (DESIGN.md 2.2) -- a real multi-session workspace carries its own history
        forward, and resetting more than a scenario explicitly declares would defeat
        the point of a longitudinal eval. `leg_start_baselines` is recomputed after
        the reset (not left as the raw-restore baselines set above) so
        `baselines.json` and every `FileUnchanged`/`FileMutated` check reflect the
        state the agent actually sees, per `scorer.py::evaluate_check`'s documented
        contract that baselines are always this leg's true start-of-leg state.
        """
        if not self.scenario.critical_residue_paths:
            return
        for rel in self.scenario.critical_residue_paths:
            seed_content = self.scenario.files[rel]  # __post_init__ guarantees presence
            target = self.workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(seed_content, encoding="utf-8")
        self.leg_start_baselines = _sha256_tree(
            self.workspace, extra_ignore=self._extra_ignore_paths()
        )
        self.record["critical_residue_reset_paths"] = list(self.scenario.critical_residue_paths)

    def start_daemon(self) -> None:
        if _port_pids(self.daemon_port):
            raise SystemExit(
                f"ABORT: port {self.daemon_port} already in use; refusing to collide "
                f"with an existing daemon"
            )
        cmd = [
            self.vectr, "start", str(self.workspace),
            "--port", str(self.daemon_port), "--memory-only",
        ]
        # Every arm except "mcp" runs with --no-ide-config (the honesty rail: no
        # vectr guidance reaches an agent unless that is the exact channel under
        # test). Arm "mcp" omits it so `vectr start`'s own internal
        # `_maybe_write_workspace_config` writes the real CLAUDE.md/.mcp.json --
        # arm "mcp" gets vectr's actual guidance artifacts, not a synthesized
        # approximation.
        if self.arm != "mcp":
            cmd.append("--no-ide-config")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=self._daemon_env()
        )
        out = f"{proc.stdout}\n{proc.stderr}"
        self._out("daemon-start.txt").write_text(f"$ {' '.join(cmd)}\nrc={proc.returncode}\n\n{out}")
        actual = _parse_started_port(out)
        if actual is not None and actual != self.daemon_port:
            print(
                f"[run_leg] daemon bound port {actual}, not the requested "
                f"{self.daemon_port}; following the real port", file=sys.stderr,
            )
            self.daemon_port = actual
        if self.daemon_port < 8899:
            raise SystemExit(
                f"ABORT: daemon landed on port {self.daemon_port}, below the 8899 "
                f"scratch floor -- refusing to talk to a possibly-live daemon"
            )
        self.record["daemon_port"] = self.daemon_port
        status = _wait_for(f"http://127.0.0.1:{self.daemon_port}/v1/status", 180.0, "scratch daemon")
        self._out("daemon-status.json").write_text(json.dumps(status, indent=2))
        self.notes_count_at_start = status.get("notes_count")
        self.record["notes_in_store_at_start"] = self.notes_count_at_start

    def _apply_hook_delivery_metadata(self, payload: dict[str, Any]) -> None:
        """Arm-conditional DELIVERY METADATA, applied at plant time only --
        DESIGN.md section 8. `variant.content`/`title` (the advisory text) is
        never touched here or anywhere else in `plant_note()`: channel parity
        across arms means every arm plants the byte-identical advisory; only
        HOW the memory layer is configured to deliver it varies per channel.

        PROBLEM 1 -- eligibility (DEFECT 6: a live hook-sessionstart leg came
        back INVALID with `hook_injection_counts: {}`, no planted content
        delivered). Root cause, empirically verified against a scratch daemon
        with `agent/trigger_engine.py` read in full: every scenario's
        `NoteVariant.trigger_paths` is non-empty, so `plant_note()` above
        always sends an EXPLICIT `triggers` list -- which, per
        `trigger_engine.effective_triggers()`'s replace-not-merge contract,
        means the note's `kind` default trigger bundle
        (`default_bundle_for_kind()`) never even applies here; kind alone is
        irrelevant to eligibility once `triggers` is explicit. The explicit
        list this harness sends is PATH-only (`{"path": glob}` per entry, no
        `event` key). `trigger_engine._trigger_matches()` requires a
        path-bearing trigger to have a non-empty `path_candidates` tuple to
        ever match -- and the SessionStart/boot delivery path
        (`app/service.py::_recall_impl`'s `boot` branch ->
        `WorkingContextStore.fire_and_format(events=["session-start"], ...)`)
        never supplies a `file_path` (there is no file being edited at boot),
        so `path_candidates` is `None` there. A path-only trigger therefore
        structurally can never fire at session-start, independent of `kind`.
        Fix: for the two arms that deliver via the real SessionStart hook
        (`hook-sessionstart`, `hook-full`), append one additional trigger
        entry, `{"event": "session-start"}`, to the SAME explicit `triggers`
        list already being sent. `evaluate_note()` composes a note's trigger
        list with OR semantics (first matching trigger in the list wins), so
        this purely ADDS session-start eligibility without touching the
        pre-existing path-anchor triggers the pre-edit/PreToolUse channel
        already relies on -- verified live against a scratch daemon
        (`vectr hook pre-tool-use` on a matching file_path still fires the
        SAME note that now also fires on `vectr hook session-start`). This
        alone (implementation option 1 of the 2 offered, `kind` left at
        "gotcha") IS honored by the shipped trigger engine -- the candidate
        finding "declared session-start triggers inert on the hook channel"
        is FALSIFIED, evidence in this commit's report.

        PROBLEM 2 -- content budget, found while verifying PROBLEM 1's fix
        end-to-end (a second, independent defect tripped over while
        implementing, per this repo's own "bugs found while building are
        product tasks" rule -- reported as a candidate UPG item, not silently
        routed around). `agent/trigger_engine.py`'s two-tier injection pack
        (`pack_injection()`) caps a fired note's FULL-text render at
        `MEMORY_TRIGGER_PER_KIND_TOKEN_CAP[kind]` before it will render in
        full; kind "gotcha" is capped at 100 tokens (`agent/config.yaml`).
        Computed directly against the shipped renderer
        (`agent.working_context_store._store._format_full_block`) for all 16
        `NoteVariant` sites in `scenarios.py`: EVERY one's full-block render
        is 101-181 tokens -- over the gotcha cap in every case, with no
        exception. A gotcha-kind hook delivery therefore ALWAYS degrades to
        its index-tier one-liner (title only, no fact content) for this
        entire scenario corpus -- so `hook_preflight()`'s (and
        `scorer.leg_non_vacuity`'s, scorer.py's arm `hook-sessionstart`
        branch) literal `planted_note_content in transcript` check can never
        pass for a gotcha-kind hook note here, independent of PROBLEM 1's
        fix. Fix: also switch `kind` to "directive" for the SAME two hook
        arms -- `MEMORY_TRIGGER_PER_KIND_TOKEN_CAP["directive"]` is 400,
        comfortably above every scenario's 101-181 token render, and
        "directive" is (like "gotcha") one of `FULL_TEXT_KINDS`, so nothing
        about the rendering FORMAT changes, only the budget headroom. Kind
        alone would not restore eligibility (an explicit `triggers` list
        still overrides the kind default per PROBLEM 1's finding), so this is
        applied ALONGSIDE, not instead of, the appended session-start
        trigger. Both fixes verified together against a live scratch daemon:
        `vectr hook session-start` returns `hook_injection_counts["SessionStart"]`
        incrementing AND the literal fact content (not just the title) inside
        `hookSpecificOutput.additionalContext`, for `bench_box_only`'s real
        "plain" variant content, unmodified.

        This is implementation option 2 of the 2 offered by the brief, but
        for a DIFFERENT reason than the brief's own fallback condition
        anticipated (trigger inertness, which PROBLEM 1's evidence
        falsifies) -- the real reason is PROBLEM 2's content-budget cap,
        which only "kind=directive" (not any triggers-list change) can clear
        for this corpus.

        Both changes are scoped to `self.arm`, a harness configuration value
        the caller (this script's own `--arm` flag) already resolved before
        planting -- not query/note content -- so neither touches the
        no-query-heuristics boundary (`agent/trigger_engine.py`'s own module
        docstring: "Nothing in this module ever reads a user prompt or query
        string"). `scenarios.py`'s 16 `NoteVariant` sites are deliberately
        left unedited; this is plant-time channel configuration only.

        Non-hook arms (`none`, `mcp`, `mcp-bare`, `proxy`) are byte-for-byte
        unaffected -- this function is a no-op for them. Arm
        "hook-userpromptsubmit" is ALSO a no-op here, despite being a hook
        arm: it delivers via `_recall_impl`'s ordinary ranked `recall()` call
        (the ungated ranked-recall pass every UserPromptSubmit request also
        makes, independent of trigger config or note kind -- see
        `probe()`'s docstring), not the SessionStart boot branch PROBLEM 1
        fixes eligibility for, so it needs neither the appended trigger nor
        the widened content budget: the planted note ships with its
        scenario-authored `kind`/`triggers`, unmodified, for this arm."""
        if self.arm not in ("hook-sessionstart", "hook-full"):
            return
        triggers = list(payload.get("triggers") or [])
        triggers.append({"event": "session-start"})
        payload["triggers"] = triggers
        payload["kind"] = "directive"

    def _reset_note_store(self) -> None:
        """UPG-EVAL-STORE-MATCH: clear every note from this trajectory's store
        immediately before this leg's plant -- `plant_note()` runs exactly once per
        trajectory (only k==2, `run_plan.py::_leg_cmd`: k>=3 forwards
        `--planted-anchor`/`--planted-note-id` instead of planting again), so the
        store must be EMPTY immediately before that single plant for the resulting
        `notes_in_store_at_start` to be note-count matched across every arm.

        Without this, a k==2 attempt that plants successfully and THEN fails later
        in the same leg (`write_hooks()`/`hook_preflight()` are extra abortable
        steps arms "hook-sessionstart"/"hook-full" run AFTER `plant_note()` that
        arm "proxy" does not) leaves its just-planted note sitting in
        `--db-dir` -- which `run_plan.py`'s failed-leg retry path
        (`_supersede(leg_dir)`) supersedes only the leg's own artifact directory,
        never the trajectory-persistent `--db-dir` itself. A retried plant then adds
        a SECOND note on top of the first attempt's, and a further retry a third,
        with no reset in between -- exactly the asymmetry a review of recorded runs
        found: hook legs beginning with 2 (k2) / 3 (k3) notes in store against
        exactly 1 in every proxy leg, because hook arms fail-after-plant more often.

        Uses the existing `/v1/memory/clear` route (`app/routes.py`, the same
        primitive `vectr_forget(all=true)` already exposes on the MCP surface) --
        not a new mechanism. Idempotent and a practical no-op on a fresh db (there
        is nothing to delete). Best-effort: a failed clear call is recorded
        diagnostically rather than aborting the leg here, because
        `verify_note_store_matched()` (called right after `plant_note()` in
        `main()`, before any paid agent session) is the actual enforcement point --
        it catches a clear that silently failed to leave the store empty via the
        post-plant count itself, with a clearer diagnostic than a raw HTTP error
        would give in isolation.
        """
        out = _http_json(
            "POST", f"http://127.0.0.1:{self.daemon_port}/v1/memory/clear", {}, timeout=30
        )
        self.record["note_store_reset_before_plant"] = out.get("deleted")

    def plant_note(self) -> None:
        self._reset_note_store()
        variant = self._find_note_variant()
        payload: dict[str, Any] = {
            "content": variant.content,
            "title": variant.title,
            "kind": variant.kind,
            "priority": variant.priority,
            "tags": list(variant.tags),
            "agent": "longitudinal-harness",
        }
        if variant.trigger_paths:
            payload["triggers"] = [{"path": p} for p in variant.trigger_paths]
        if variant.anchors:
            payload["anchors"] = list(variant.anchors)
        self._apply_hook_delivery_metadata(payload)
        out = _http_json(
            "POST", f"http://127.0.0.1:{self.daemon_port}/v1/remember", payload, timeout=60
        )
        nid = out.get("note_id")
        if not isinstance(nid, int):
            raise SystemExit(f"ABORT: could not plant note: {out}")
        self.note_id = nid
        self.planted_anchor = f"note:{nid}"
        self.record["planted_note_id"] = nid
        self.record["planted_anchor"] = self.planted_anchor

        # `notes_count_at_start` (set in start_daemon(), BEFORE this plant) is the
        # scorer's non-vacuity premise counter for the memory arms
        # (scorer.leg_non_vacuity's "notes_count_at_start is 0" checks) -- its
        # documented premise is "note in store when the AGENT starts", and the
        # agent starts AFTER this plant for every --plant-note leg. The pre-plant
        # snapshot is therefore stale for this leg specifically; replace it with a
        # fresh post-plant count so the recorded premise matches what the agent
        # will actually see. The stale pre-plant value is preserved separately
        # under a diagnostic key rather than discarded. A failed re-snapshot
        # (daemon hiccup) falls back to keeping the pre-plant value rather than
        # aborting the leg over a diagnostic read.
        self.record["notes_in_store_pre_plant"] = self.notes_count_at_start
        post_plant_status = _http_json(
            "GET", f"http://127.0.0.1:{self.daemon_port}/v1/status", timeout=30
        )
        post_plant_count = post_plant_status.get("notes_count")
        if isinstance(post_plant_count, int):
            self.notes_count_at_start = post_plant_count
            self.record["notes_in_store_at_start"] = post_plant_count

    def verify_note_store_matched(self) -> None:
        """UPG-EVAL-STORE-MATCH pre-spend gate: mirrors `probe()`'s and
        `hook_preflight()`'s "fix the scenario, not the score" contract -- catch a
        store-count mismatch BEFORE the paid `claude -p` session, not after.

        Called from `main()` right after the `--plant-note` step (whether or not
        THIS leg itself planted -- k>=3 legs never plant but still read this same
        trajectory-persistent store) and before any daemon-side probe or agent
        spawn. For arm "none" there is nothing to check (its own branch elsewhere
        already asserts an empty store). For every arm except
        `_ARMS_WITH_LIVE_TOOL_ACCESS`, the store's only possible content-adding
        event across an entire trajectory is this harness's own single
        `plant_note()` call at k==2 (`_reset_note_store()`'s docstring has the
        retry-accumulation defect this closes) -- so `notes_count_at_start` must be
        exactly 1, never merely non-zero. `_ARMS_WITH_LIVE_TOOL_ACCESS` arms are
        exempt: their agent has live vectr tool access, so a count above 1 by a
        later leg can be genuine agent behavior from an earlier leg, not a defect.

        Aborts (nonzero exit) on a mismatch unless `--allow-store-mismatch` is
        passed, matching every other preflight gate's override pattern in this
        file. `scorer.leg_non_vacuity()` carries the same invariant for the
        arms this function also gates (post-hoc, arm-blind-outcome-preserving) --
        this function exists so the mismatch is caught before money is spent, not
        only recorded afterward.
        """
        if self.arm == "none" or self.arm in _ARMS_WITH_LIVE_TOOL_ACCESS:
            return
        if self.notes_count_at_start == 1:
            return
        self.record["store_count_matched"] = False
        if self.args.allow_store_mismatch:
            self.record["store_mismatch_allowed"] = True
            return
        raise SystemExit(
            f"ABORT: notes_in_store_at_start={self.notes_count_at_start!r} != 1 for "
            f"arm {self.arm!r} at k={self.k} (UPG-EVAL-STORE-MATCH) -- this arm's "
            f"agent has no live vectr tool access, so the store's only "
            f"content-adding event is this harness's own single plant at k==2; any "
            f"other count means the trajectory-persistent --db-dir is carrying "
            f"stray notes from an earlier failed-then-retried attempt at this same "
            f"k, or the plant/forward chain broke. Pass --allow-store-mismatch to "
            f"override for debugging."
        )

    def probe(self) -> None:
        """Daemon-side reachability probes (DESIGN.md 4.1, 7.3, and the
        UPG-EVAL-PLANT-DISPLACEMENT probe rewrite). Never touches the proxy --
        see `_proactive_probe`'s docstring.

        The reachability ABORT criterion is arm-aware (DEFECT 7). `_proactive_probe`
        below queries the daemon's proactive/proxy channel (`/v1/proactive`,
        `channel="proxy"`) -- but for arms "hook-sessionstart"/"hook-full",
        `plant_note()` plants the note with `kind="directive"`
        (`_apply_hook_delivery_metadata`, required to fit the SessionStart
        channel's per-kind content-injection budget -- see that method's
        docstring). `agent/config.yaml`'s `proactive.structural_kinds` omits
        "directive" outright, and `proactive.proxy.exclude_directive_notes`
        independently excludes it from this channel too (a SEPARATE toggle, for
        a different reason -- authority confusion from injecting an imperative
        into a live user turn -- see that key's own docstring). A directive-kind
        note is therefore structurally unable to ever appear in
        `_proactive_probe`'s result, regardless of whether the SessionStart
        channel it actually ships through can see it -- querying the wrong
        channel as the reachability gate aborted a live T2 run at $0 even
        though `hook_preflight()` (run later, once hooks are installed) already
        proves session-start delivery works end to end for the same note.
        `_proactive_probe` is still ALWAYS queried and recorded here (a
        diagnostic-only value for hook arms -- `preflight["proactive_probe_
        diagnostic_only"]` marks this), but hook arms judge reachability
        against the SAME channel their leg actually tests: `_session_start_probe`
        below issues the identical `boot=True, hook_event="SessionStart"`
        `/v1/recall` payload `agent/hook_cli.py::run_hook`'s own "session-start"
        branch sends, directly against the daemon -- channel-true and
        zero-LLM-cost, and it does not require `write_hooks()` to have run yet
        (this method runs before hooks are installed -- see `main()`'s call
        order).

        UPG-EVAL-PLANT-DISPLACEMENT (2026-08-03 ruling): a live k>=2 leg
        aborted pre-spend because the agent's OWN note from a prior leg (an
        anchored, high-priority gotcha) legitimately outranked the unanchored
        planted directive at the proactive channel's default item budget.
        That ranking is working as intended -- an agent-authored, anchored,
        high-priority note SHOULD be able to win a proactive slot -- so
        treating "displaced" the same as "absent" was the actual defect:
        every mcp/proxy trajectory whose agent writes a strong note would
        otherwise self-terminate at its very next leg. Non-hook memory arms
        (mcp, mcp-bare, proxy) therefore now gate on a DIRECT by-id
        existence/integrity check (`_note_by_id_probe`, `/v1/recall` with
        `note_id` set -- independent of any channel's ranking) as the
        PRIMARY reachability test; `_proactive_probe`'s result is
        corroborating diagnostics, never the gate, for these arms. When the
        by-id check passes but the note is absent from the channel's
        default-budget response, `_proactive_rank_probe` locates its actual
        rank (or confirms it never surfaces within budget) and the leg
        records `planted_rank`/`displaced_by`/`delivered_at_default`/
        `channel_delivery` and RUNS -- "the channel fails to deliver under
        contention" is a genuine measured property of THIS run, not an
        instrument error (`scorer.leg_non_vacuity`'s arm "proxy" branch
        reads `channel_delivery` so it does not then independently
        re-invalidate the same already-accepted leg via its own post-hoc
        delivery expectation). ABORT is now reserved for genuine
        unreachability: the by-id check finds the note absent or
        content-corrupt (`preflight["reachable_channel"] == "by_id"`), or
        every probe path fails at the transport level -- daemon unreachable,
        or the embedder never comes up within the polling window
        (`preflight["infra_unreachable"] = True`, checked via `_transport_ok`
        on the by-id probe AND both proactive-probe turns). Hook arms are
        unaffected by any of this: each judges reachability against its own
        channel-true probe (SessionStart immediately below; the
        UserPromptSubmit probe for arm "hook-userpromptsubmit"). For the
        SessionStart arms a directive-kind note is structurally excluded
        from the proactive channel's ranking in the first place, so this
        specific displacement cannot occur there. The UserPromptSubmit probe
        DOES check planted content in an ordinary ranked `/v1/recall`
        response, so at high k an agent-note-rich store could in principle
        displace the planted note out of the recall window -- if a USPS leg
        ever aborts on that probe at k>=2, extend this same displacement
        tolerance to that channel before spending, rather than raising the
        recall limit.

        Arm "hook-userpromptsubmit" is a THIRD case, distinct from both:
        `plant_note()` does NOT force it to `kind="directive"` (see
        `_apply_hook_delivery_metadata`'s own arm check, unchanged), because
        this arm tests the per-prompt semantic-recall path (`VectrService.
        _recall_impl`'s ordinary ranked `recall()` call, un-gated by note kind
        or trigger config -- every planted note is naturally eligible there,
        unlike SessionStart's boot-only call). It still never uses
        `_proactive_probe` as its abort criterion, because `start_proxy()`
        passes `--no-inject` for every arm except "proxy" -- so even a
        non-directive note visible to `/v1/proactive` in principle never
        actually ships through that channel during the real session for this
        arm either. `_user_prompt_submit_probe` below issues the identical
        `{"query": ..., "hook_event": "UserPromptSubmit", "events":
        ["prompt-submit"]}` `/v1/recall` payload `agent/hook_cli.py::
        run_hook`'s own "user-prompt-submit" branch sends, using the leg's own
        prompt as the query -- channel-true for the semantic-recall path this
        arm actually exercises.
        """
        memory_arm = self.arm != "none"
        if memory_arm and (self.note_id is not None or self.planted_anchor):
            variant = self._find_note_variant()
            turn1, turn2 = _proactive_probe(
                self.daemon_port, self.leg_spec, self.workspace,
                f"longprobe-{self.scenario.slug}-k{self.k}",
            )
            anchor = self.planted_anchor
            preflight = {
                "turn1_text_only": {
                    "item_count": turn1.get("item_count"),
                    "chars": len(turn1.get("context") or ""),
                    "anchor_ids": turn1.get("anchor_ids"),
                    "planted_present": anchor in (turn1.get("anchor_ids") or []),
                },
                "turn2_with_file_anchor": {
                    "item_count": turn2.get("item_count"),
                    "chars": len(turn2.get("context") or ""),
                    "anchor_ids": turn2.get("anchor_ids"),
                    "planted_present": anchor in (turn2.get("anchor_ids") or []),
                },
            }
            proactive_reachable = (
                preflight["turn1_text_only"]["planted_present"]
                or preflight["turn2_with_file_anchor"]["planted_present"]
            )

            is_session_start_hook_arm = self.arm in ("hook-sessionstart", "hook-full")
            is_usps_hook_arm = self.arm == "hook-userpromptsubmit"
            is_hook_arm = is_session_start_hook_arm or is_usps_hook_arm
            if is_session_start_hook_arm:
                session_start_notes = _session_start_probe(
                    self.daemon_port, f"longprobe-{self.scenario.slug}-k{self.k}-sessionstart",
                )
                session_start_reachable = bool(variant.content) and (
                    _collapse_ws(variant.content) in _collapse_ws(session_start_notes)
                )
                preflight["session_start_channel"] = {
                    "reachable": session_start_reachable,
                    "notes_chars": len(session_start_notes),
                }
                preflight["proactive_probe_diagnostic_only"] = True
                preflight["reachable_channel"] = "session_start"
                reachable = session_start_reachable
            elif is_usps_hook_arm:
                user_prompt_submit_notes = _user_prompt_submit_probe(
                    self.daemon_port,
                    f"longprobe-{self.scenario.slug}-k{self.k}-userpromptsubmit",
                    self.leg_spec.prompt,
                )
                user_prompt_submit_reachable = bool(variant.content) and (
                    _collapse_ws(variant.content) in _collapse_ws(user_prompt_submit_notes)
                )
                preflight["user_prompt_submit_channel"] = {
                    "reachable": user_prompt_submit_reachable,
                    "notes_chars": len(user_prompt_submit_notes),
                }
                preflight["proactive_probe_diagnostic_only"] = True
                preflight["reachable_channel"] = "user_prompt_submit"
                reachable = user_prompt_submit_reachable
            else:
                preflight["proactive_probe_diagnostic_only"] = False
                preflight["reachable_channel"] = "by_id"
                by_id = (
                    _note_by_id_probe(self.daemon_port, self.note_id, variant.content)
                    if self.note_id is not None
                    else {"transport_ok": False, "exists": False, "content_matches": False, "raw_chars": 0}
                )
                preflight["by_id_probe"] = by_id
                integrity_ok = by_id["exists"] and (by_id["content_matches"] or not variant.content)
                reachable = integrity_ok

                if integrity_ok:
                    if proactive_reachable:
                        self.record["delivered_at_default"] = True
                        self.record["channel_delivery"] = "delivered_default"
                    else:
                        cap = max(1, min(self.notes_count_at_start or 1, 10))
                        rank_probe = _proactive_rank_probe(
                            self.daemon_port, self.leg_spec, self.workspace,
                            f"longprobe-{self.scenario.slug}-k{self.k}", anchor, cap,
                        )
                        preflight["rank_probe"] = rank_probe
                        self.record["planted_rank"] = rank_probe["rank"]
                        self.record["displaced_by"] = rank_probe["displaced_by"]
                        self.record["delivered_at_default"] = False
                        self.record["channel_delivery"] = "displaced"
                else:
                    self.record["channel_delivery"] = "unreachable"
                    preflight["infra_unreachable"] = (
                        not by_id["transport_ok"]
                        and not _transport_ok(turn1) and not _transport_ok(turn2)
                    )

            self._out("preflight.json").write_text(json.dumps(preflight, indent=2))
            self.record["preflight"] = preflight
            self.record["planted_note_reachable_preflight"] = reachable

            if variant.variant in ("provenance", "verifiable"):
                trail_text = variant.content.replace(self.scenario.fact_sentence, "", 1).strip()
                # DEFECT 8: normalize BOTH sides -- delivery collapses whitespace
                # (_collapse_ws docstring above), so an authored trail with an
                # internal newline (e.g. S1-verifiable's provenance sentence) must
                # be collapsed identically before the containment check, or a
                # multi-line trail false-negatives even when delivery succeeded.
                trail_text_collapsed = _collapse_ws(trail_text)
                found = bool(trail_text) and (
                    trail_text_collapsed in _collapse_ws(turn1.get("context") or "")
                    or trail_text_collapsed in _collapse_ws(turn2.get("context") or "")
                )
                self.trail_text_delivered = found
                self.record["trail_text_delivered"] = found

            if self.arm in ("mcp", "mcp-bare"):
                # DEFECT 13 (1/2): a pre-session MCP streamable-HTTP handshake
                # against the daemon's real `/mcp` route -- proves the server this
                # leg's `claude -p` session is about to connect to actually serves
                # the vectr tool surface, independent of `system.init`'s
                # `mcp_servers` status (see `_mcp_handshake_probe`'s docstring).
                handshake = _mcp_handshake_probe(self.daemon_port)
                self._out("mcp-handshake.json").write_text(json.dumps(handshake, indent=2))
                self.mcp_handshake_ok = handshake["ok"]
                self.record["mcp_handshake_ok"] = handshake["ok"]
                self.record["mcp_handshake_tools"] = handshake["tool_count"]
                if not handshake["ok"] and not self.args.allow_unreachable:
                    raise SystemExit(
                        "ABORT: the MCP streamable-HTTP handshake against the "
                        f"daemon's /mcp route (port {self.daemon_port}) did not "
                        "return a vectr_* tool from tools/list -- the server this "
                        "leg's claude -p session is about to connect to is not "
                        "actually serving the vectr tool surface. Fix the daemon "
                        "(not the score); re-run with --allow-unreachable to "
                        "record it anyway."
                    )

                # DEFECT 13 (2/2): poll (not single-shot) so an embedder still
                # warming up at probe time isn't mistaken for an unreachable
                # channel -- see `_poll_recall_probe`'s docstring.
                recall_probe = _poll_recall_probe(
                    self.daemon_port, self.leg_spec.prompt, self.note_id,
                )
                self.recall_probe_returned_note = recall_probe["returned"]
                self.record["recall_probe_returned_note"] = recall_probe["returned"]
                self.record["recall_probe_method"] = recall_probe["method"]
                self.record["recall_probe_elapsed_s"] = recall_probe["elapsed_s"]

            if not reachable and not self.args.allow_unreachable:
                if is_session_start_hook_arm or is_usps_hook_arm:
                    channel = (
                        "the SessionStart channel" if is_session_start_hook_arm
                        else "the UserPromptSubmit channel"
                    )
                    raise SystemExit(
                        f"ABORT: the planted note is not retrievable on {channel}, "
                        "so this leg cannot test its memory channel. Fix the "
                        "scenario (not the score); re-run with --allow-unreachable "
                        "to record it anyway."
                    )
                if preflight.get("infra_unreachable"):
                    raise SystemExit(
                        "ABORT: infra unreachable -- neither the by-id /v1/recall "
                        "integrity probe nor the daemon-side proactive probes got "
                        f"a transport-level response from the daemon (port "
                        f"{self.daemon_port}), so this leg's memory channel "
                        "cannot be tested at all. Fix the daemon (not the score); "
                        "re-run with --allow-unreachable to record it anyway."
                    )
                raise SystemExit(
                    "ABORT: the planted note is absent or content-corrupt in the "
                    "daemon's store (by-id /v1/recall integrity check failed), so "
                    "this leg cannot test its memory channel. Fix the scenario "
                    "(not the score); re-run with --allow-unreachable to record "
                    "it anyway."
                )

        # Everything written up to here is this leg's own preflight traffic; wait
        # for the audit log to settle so non-vacuity counts only what follows
        # (mirrors injection_utility/run_harness.py's `_settled_audit_size`).
        self.audit_offset = self._settled_audit_size()
        self.record["audit_offset_after_preflight"] = self.audit_offset

    def _settled_audit_size(self, *, quiet_s: float = 0.6, limit_s: float = 8.0) -> int:
        deadline = time.time() + limit_s
        last = self.audit_log.stat().st_size if self.audit_log.exists() else 0
        while time.time() < deadline:
            time.sleep(quiet_s)
            now = self.audit_log.stat().st_size if self.audit_log.exists() else 0
            if now == last:
                return now
            last = now
        return last

    def start_proxy(self) -> None:
        self.proxy_port = _free_port_at_or_above(max(self.proxy_port, 8899), avoid={self.daemon_port})
        self.record["proxy_port"] = self.proxy_port
        cmd = [
            self.vectr, "proxy",
            "--port", str(self.proxy_port),
            "--daemon-port", str(self.daemon_port),
            "--path", str(self.workspace),
        ]
        # Only arm "proxy" (arm C) has injection enabled -- DESIGN.md 4. Every
        # other arm, including the hook arms, keeps a transparent proxy on the
        # path (see module docstring).
        if self.arm != "proxy":
            cmd.append("--no-inject")
        stderr_path = self._out("proxy.stderr")
        self._proxy_stderr = stderr_path.open("w")
        self.proxy_proc = subprocess.Popen(
            cmd, stdout=self._proxy_stderr, stderr=subprocess.STDOUT, env=self._daemon_env()
        )
        health = _wait_for(f"http://127.0.0.1:{self.proxy_port}/__vectr_proxy/health", 90.0, "scratch proxy")
        self.record["proxy_instance_id"] = health.get("instance_id")
        self.record["proxy_command"] = " ".join(cmd)

    def write_hooks(self) -> None:
        """Install Claude Code hook entries for arms "hook-sessionstart"/
        "hook-full"/"hook-userpromptsubmit".

        `vectr init --hooks` has no CLI-level granularity to install a single hook
        group (main.py's `_write_claude_hooks` always writes all six events) --
        for "hook-sessionstart" (D1), the full set is installed and then this
        method deletes every `.claude/settings.json` hook key except
        "SessionStart" directly; for "hook-userpromptsubmit" it prunes to
        "UserPromptSubmit" instead, the same way -- isolating the per-prompt
        semantic-recall channel as its own arm, with no SessionStart and no
        PreToolUse hook wired at all. This is a scratch-workspace harness artifact
        edit, the same category as arm "mcp-bare"'s hand-built mcp.json -- not a
        product-code change and not a workaround for a vectr defect.
        """
        cmd = [self.vectr, "init", "--path", str(self.workspace), "--hooks", "--no-ide-config"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=self._daemon_env())
        out = f"{proc.stdout}\n{proc.stderr}"
        self._out("init-hooks.txt").write_text(f"$ {' '.join(cmd)}\nrc={proc.returncode}\n\n{out}")
        settings_path = self.workspace / ".claude" / "settings.json"
        if proc.returncode != 0 or not settings_path.exists():
            raise SystemExit(
                f"ABORT: `vectr init --hooks` did not produce {settings_path} "
                f"(rc={proc.returncode}); see {self._out('init-hooks.txt')}"
            )
        prune_to: str | None = None
        if self.arm == "hook-sessionstart":
            prune_to = "SessionStart"
        elif self.arm == "hook-userpromptsubmit":
            prune_to = "UserPromptSubmit"
        if prune_to is not None:
            data = json.loads(settings_path.read_text())
            hooks = data.get("hooks") or {}
            for event in list(hooks):
                if event != prune_to:
                    del hooks[event]
            data["hooks"] = hooks
            settings_path.write_text(json.dumps(data, indent=2))
            self.record["hooks_pruned_to"] = [prune_to]
        self.record["hooks_settings_path"] = str(settings_path)

    def _agent_cwd(self) -> Path:
        """The cwd both the spawned `claude` process (`run_agent()`) and the
        synthetic hook invocation (`hook_preflight()`) use -- one function so the
        two can never drift apart (the hook preflight's whole point is to prove
        the mechanism under the SAME conditions the real agent session will see).
        """
        return self.workspace if self.scenario.agent_cwd == "." else self.workspace / self.scenario.agent_cwd

    def _session_start_hook_command(self) -> str:
        """The `command` string `.claude/settings.json`'s SessionStart hook group
        carries -- `write_hooks()` must have already run. `vectr init --hooks`
        (main.py's `_install_hook_group`) always writes exactly one vectr-managed
        entry per event, so the first command found is the only one there is.
        """
        settings_path = self.workspace / ".claude" / "settings.json"
        if not settings_path.is_file():
            raise SystemExit(
                f"ABORT: hook preflight found no {settings_path} -- write_hooks() "
                f"must run before hook_preflight()"
            )
        data = json.loads(settings_path.read_text())
        groups = (data.get("hooks") or {}).get("SessionStart") or []
        commands = [
            h.get("command")
            for group in groups
            for h in (group.get("hooks") or [])
            if isinstance(h, dict) and h.get("command")
        ]
        if not commands:
            raise SystemExit(
                f"ABORT: {settings_path} has no SessionStart hook command -- "
                f"write_hooks() should have installed one for arm {self.arm!r}"
            )
        return commands[0]

    def hook_preflight(self) -> None:
        """Zero-cost SessionStart hook mechanism preflight -- the hook-arm analogue
        of `probe()`'s daemon-side reachability check, run BEFORE the paid
        `claude -p` session for arms "hook-sessionstart"/"hook-full" (mirrors
        `probe()`'s "fix the scenario, not the score" contract exactly).

        Executes the workspace's OWN configured SessionStart hook command exactly
        as `claude` would invoke it: same cwd (`_agent_cwd()`), same env as the
        spawned agent subprocess (`_spawn_env_for_agent`), and a synthetic Claude
        Code SessionStart hook stdin payload carrying every field
        `agent/hook_cli.py`'s "session-start" branch reads (`cwd`, `session_id`,
        `source`) plus the schema fields Claude Code's own hook input always
        includes (`transcript_path`, `hook_event_name`) for realism.

        Asserts BOTH:
          (i) daemon-side evidence: THIS leg's own scratch daemon
              (`self.daemon_port` -- never any other daemon on the box) shows its
              `/v1/status` `hook_injection_counts["SessionStart"]` counter
              increment across the call. This is what structurally catches
              instance mis-resolution too -- if the hook command resolves (via the
              global `~/.vectr/instances.json` registry) to a DIFFERENT daemon
              than this leg's own, the counter on THIS daemon never moves and the
              preflight aborts, exactly as it should.
          (ii) content evidence: the hook's own stdout carries the planted note's
              content inside its `hookSpecificOutput.additionalContext` envelope
              (`_content_delivered_in_json_text`, DEFECT 8 -- whitespace-
              normalized AND JSON-escape-aware, since stdout is the raw
              `print(json.dumps(...))` output, not the decoded field value; a
              multi-line variant's internal newline is JSON-escaped there --
              the same check `scorer.leg_non_vacuity` already applies post-hoc
              to the real transcript for arm "hook-sessionstart"; here it is
              applied pre-hoc to the hook's own output, before any money is
              spent).

        Either failing aborts (nonzero exit) unless `--allow-hook-unreachable` is
        passed. Full detail (resolved command, stdout/stderr, before/after
        counters, registry diagnostic) is written to `hook-preflight.json`
        regardless of outcome.

        This can only ever prove the MECHANISM works using the harness's own
        env/PATH (registry resolution, hook stdin parsing, delivery content) -- it
        cannot prove `claude` itself trusts this project's `.claude/settings.json`
        from a fresh/untrusted directory when IT spawns the hook subprocess; that
        is a loading-layer question `run_agent()`'s `--settings` flag addresses,
        not provable here without spending a real `claude -p` session.
        """
        if self.arm not in ("hook-sessionstart", "hook-full"):
            return
        if not (self.note_id is not None or self.planted_anchor):
            return  # nothing planted on this leg yet -- nothing to preflight

        command = self._session_start_hook_command()
        cwd = self._agent_cwd()
        variant = self._find_note_variant()

        before = _http_json("GET", f"http://127.0.0.1:{self.daemon_port}/v1/status", timeout=10)
        before_counts = dict(before.get("hook_injection_counts") or {})

        stdin_payload = json.dumps({
            "session_id": f"hookpreflight-{self.scenario.slug}-k{self.k}",
            "transcript_path": str(self._out("hook-preflight-transcript.jsonl")),
            "cwd": str(cwd),
            "hook_event_name": "SessionStart",
            "source": "startup",
        })

        base_url = f"http://127.0.0.1:{self.proxy_port}"
        started = time.time()
        rc, stdout, stderr, timed_out = _run_hook_command(
            command, cwd=str(cwd), stdin=stdin_payload,
            env=_spawn_env_for_agent(base_url), timeout=60.0,
        )
        elapsed = round(time.time() - started, 3)

        after = _http_json("GET", f"http://127.0.0.1:{self.daemon_port}/v1/status", timeout=10)
        after_counts = dict(after.get("hook_injection_counts") or {})
        delta = int(after_counts.get("SessionStart", 0) or 0) - int(before_counts.get("SessionStart", 0) or 0)
        daemon_evidence = delta > 0

        # DEFECT 8: stdout is the hook subprocess's raw JSON-encoded print
        # output (`_emit_hook_context`), not the decoded `additionalContext`
        # text -- a literal or merely whitespace-collapsed containment check
        # false-negatives on a content string with an internal newline (JSON-
        # escaped to `\n` in stdout). See `_content_delivered_in_json_text`.
        stdout_has_content = _content_delivered_in_json_text(variant.content, stdout or "")

        registry_port = _registry_port_for(cwd)

        result = {
            "command": command,
            "cwd": str(cwd),
            "rc": rc,
            "timed_out": timed_out,
            "elapsed_s": elapsed,
            "stdout": stdout,
            "stderr": stderr,
            "hook_injection_counts_before": before_counts,
            "hook_injection_counts_after": after_counts,
            "session_start_delta": delta,
            "daemon_evidence": daemon_evidence,
            "stdout_has_planted_content": stdout_has_content,
            "leg_daemon_port": self.daemon_port,
            "registry_resolved_port": registry_port,
            "registry_port_matches_leg_daemon": (
                (registry_port == self.daemon_port) if registry_port is not None else None
            ),
        }
        preflight_path = self._out("hook-preflight.json")
        preflight_path.write_text(json.dumps(result, indent=2))
        self.record["hook_preflight"] = result

        ok = daemon_evidence and stdout_has_content
        if not ok and not self.args.allow_hook_unreachable:
            mismatch = ""
            if registry_port is not None and registry_port != self.daemon_port:
                mismatch = (
                    f"; the instance registry resolves this cwd to port "
                    f"{registry_port}, not this leg's own daemon port "
                    f"{self.daemon_port} -- likely instance mis-resolution"
                )
            raise SystemExit(
                "ABORT: SessionStart hook preflight failed on this leg's own "
                f"scratch daemon (port {self.daemon_port}): daemon_evidence="
                f"{daemon_evidence} (hook_injection_counts['SessionStart'] delta="
                f"{delta}) stdout_has_planted_content={stdout_has_content}{mismatch}. "
                "This leg cannot test its hook channel -- fix the scenario (not "
                "the score); re-run with --allow-hook-unreachable to record it "
                f"anyway. See {preflight_path}."
            )

        # UPG-EVAL-HOOKCOUNT-CUMULATIVE: `probe()` (run strictly before this
        # method -- see `main()`'s call order) already settled `self.
        # audit_offset` once, but that settle predates the hook subprocess
        # call just above, which is itself this leg's own preflight traffic.
        # Re-settle now so the "post-offset" window `scorer.leg_non_vacuity`
        # reads via `audit_since_offset` starts at the real `run_agent()`
        # session that follows, excluding this method's own traffic too (not
        # just `probe()`'s) -- same audit-log-quiescence mechanism `probe()`
        # itself uses, see its own comment there.
        self.audit_offset = self._settled_audit_size()
        self.record["audit_offset_after_preflight"] = self.audit_offset

    def _mcp_config_path(self) -> Path | None:
        """Path to pass via `--mcp-config` for the two MCP arms, else None (no
        config file at all; `--strict-mcp-config` alone then yields zero servers).

        Arm "mcp" uses vectr's OWN auto-written `<workspace>/.mcp.json` (from
        `start_daemon` omitting `--no-ide-config`) -- the real artifact a user
        gets, not a synthesized approximation. Arm "mcp-bare" (tools, no CLAUDE.md
        guidance) keeps `--no-ide-config`, so nothing is auto-written; this hand-
        builds the identical template `main.py` uses (`{"mcpServers": {"vectr":
        {"type": "http", "url": ...}}}`) into `artifacts/mcp.json`, OUTSIDE the
        workspace so it never pollutes workspace state/manifests.
        """
        if self.arm == "mcp":
            p = self.workspace / ".mcp.json"
            if not p.is_file():
                raise SystemExit(
                    f"ABORT: arm 'mcp' expected vectr to auto-write {p} "
                    f"(start_daemon omits --no-ide-config for this arm) but it is missing"
                )
            return p
        if self.arm == "mcp-bare":
            p = self._out("mcp.json")
            p.write_text(json.dumps({
                "mcpServers": {"vectr": {"type": "http", "url": f"http://localhost:{self.daemon_port}/mcp"}}
            }, indent=2))
            return p
        return None

    # -- the measured run -----------------------------------------------

    def run_agent(self) -> None:
        claude = shutil.which("claude") or "claude"
        cwd = self._agent_cwd()
        cmd = [
            claude, "-p", self.leg_spec.prompt,
            "--output-format", "stream-json", "--verbose",
            "--model", self.args.model,
            "--max-turns", str(self.args.max_turns),
            "--dangerously-skip-permissions",
            "--strict-mcp-config",
        ]
        if self.arm in ("hook-sessionstart", "hook-full", "hook-userpromptsubmit"):
            # Headless `claude -p` loading PROJECT-level `.claude/settings.json`
            # hooks depends on directory trust, which a freshly-materialized
            # scratch workspace never has (`--dangerously-skip-permissions` skips
            # tool-call approval, not directory trust). `--settings` loads the
            # file explicitly regardless of trust state, so hook delivery no
            # longer depends on it. Defensive/additive: `write_hooks()` already
            # ran by the time this executes (main()'s call order), so
            # hooks_settings_path is always set for these three arms.
            cmd += ["--settings", self.record["hooks_settings_path"]]
        mcp_config = self._mcp_config_path()
        if mcp_config is not None:
            cmd += ["--mcp-config", str(mcp_config)]
        transcript = self._out("transcript.jsonl")
        base_url = f"http://127.0.0.1:{self.proxy_port}"

        # DEFECT 7/DESIGN.md 4.1 style: arm "hook-userpromptsubmit" has no
        # SessionStart or PreToolUse hook wired at all, and its
        # additionalContext never renders into the stream-json transcript
        # (UPG-IU-HOOK-NONVACUITY-CANARY) -- so, unlike D1's
        # `hook_preflight()` (a synthetic pre-spend hook invocation),
        # firing/delivery evidence for this arm can only come from the real
        # agent session's own effect on the daemon's cumulative
        # `hook_injection_counts["UserPromptSubmit"]` counter. Capture it
        # immediately before and after THIS subprocess (not the whole leg --
        # `capture_and_teardown()`'s later snapshot would also include this
        # leg's own preflight traffic) and record the delta for
        # `scorer.leg_non_vacuity` (never transcript content for this arm).
        #
        # UPG-EVAL-HOOKCOUNT-CUMULATIVE: arms "hook-sessionstart"/"hook-full"
        # bracket the SAME subprocess the same way, for the same reason --
        # `hook_preflight()` (run just before this method, D1 only) and
        # `probe()`'s `_session_start_probe()` (run even earlier, every hook
        # arm) both legitimately fire the daemon's `hook_injection_counts`
        # counters as their own pre-spend mechanism/reachability checks, so
        # the end-of-leg CUMULATIVE snapshot `capture_and_teardown()` used to
        # read directly overstated what THIS leg's real agent session alone
        # caused. `hook_fired = any(... > 0 ...)` in `scorer.leg_non_vacuity`
        # would otherwise read True purely from `hook_preflight()`'s own
        # passing check, independent of whether the real session's hook
        # fired at all -- most consequential for arm "hook-full", whose
        # non-vacuity gate has no transcript-content conjunct to fall back
        # on (see that function's own arm branch).
        usps_count_before: int | None = None
        hook_counts_before: dict[str, Any] | None = None
        if self.arm in ("hook-sessionstart", "hook-full", "hook-userpromptsubmit"):
            before = _http_json("GET", f"http://127.0.0.1:{self.daemon_port}/v1/status", timeout=10)
            hook_counts_before = dict(before.get("hook_injection_counts") or {})
            if self.arm == "hook-userpromptsubmit":
                usps_count_before = int(hook_counts_before.get("UserPromptSubmit", 0) or 0)

        started = time.time()
        with transcript.open("w") as out:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), stdout=out, stderr=subprocess.PIPE, text=True,
                env=_spawn_env_for_agent(base_url),
            )
            try:
                _, stderr = proc.communicate(timeout=self.args.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, stderr = proc.communicate()
                stderr = (stderr or "") + f"\n[run_leg] timed out after {self.args.timeout_s}s"
        self._out("agent.stderr").write_text(stderr or "")
        self.record["agent_returncode"] = proc.returncode
        self.record["agent_wall_s"] = round(time.time() - started, 1)

        if self.arm in ("hook-sessionstart", "hook-full", "hook-userpromptsubmit"):
            after = _http_json("GET", f"http://127.0.0.1:{self.daemon_port}/v1/status", timeout=10)
            hook_counts_after = dict(after.get("hook_injection_counts") or {})
            if self.arm == "hook-userpromptsubmit":
                usps_count_after = int(hook_counts_after.get("UserPromptSubmit", 0) or 0)
                self.record["user_prompt_submit_injection_count_before"] = usps_count_before
                self.record["user_prompt_submit_injection_count_after"] = usps_count_after
                self.record["user_prompt_submit_injection_delta"] = usps_count_after - (usps_count_before or 0)
            else:
                keys = set(hook_counts_before or {}) | set(hook_counts_after)
                self.record["hook_injection_counts_before_agent"] = hook_counts_before
                self.record["hook_injection_counts_after_agent"] = hook_counts_after
                self.record["hook_injection_counts"] = {
                    k: int(hook_counts_after.get(k, 0) or 0) - int((hook_counts_before or {}).get(k, 0) or 0)
                    for k in keys
                }

    # -- teardown + scoring ------------------------------------------------

    def capture_and_teardown(self) -> None:
        if self.arm in ("hook-sessionstart", "hook-full", "hook-userpromptsubmit"):
            status = _http_json("GET", f"http://127.0.0.1:{self.daemon_port}/v1/status", timeout=10)
            self._out("daemon-status-final.json").write_text(json.dumps(status, indent=2))
            # UPG-EVAL-HOOKCOUNT-CUMULATIVE: this final `/v1/status` read is
            # end-of-leg CUMULATIVE (it also includes this leg's own pre-agent
            # `hook_preflight()`/`probe()` traffic on top of the real agent
            # session) -- kept here purely as a diagnostic snapshot, never as
            # the non-vacuity evidence `scorer.leg_non_vacuity` consumes.
            # `run_agent()` already recorded the delta-bracketed
            # `record["hook_injection_counts"]` (arms "hook-sessionstart"/
            # "hook-full") and `record["user_prompt_submit_injection_delta"]`
            # (arm "hook-userpromptsubmit") that scoring actually reads; do
            # not overwrite either here.
            self.record["hook_injection_counts_cumulative_final"] = status.get("hook_injection_counts")

        health = _http_json("GET", f"http://127.0.0.1:{self.proxy_port}/__vectr_proxy/health", timeout=10)
        self._out("proxy-health.json").write_text(json.dumps(health, indent=2))
        self.record["proxy_metrics"] = health.get("metrics")

        if self.proxy_proc is not None:
            self.proxy_proc.terminate()
            try:
                self.proxy_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proxy_proc.kill()
        if self._proxy_stderr is not None:
            try:
                self._proxy_stderr.close()
            except Exception:
                pass
        self.record["proxy_hard_killed"] = _hard_stop_port(self.proxy_port)
        self.record["daemon_hard_killed"] = _stop_daemon(
            self.vectr, self.workspace, self.daemon_port, self._daemon_env()
        )

    def score(self) -> None:
        scen.materialize_verifiers(self.leg_spec, self.verify_dir)
        events = scorer.load_transcript(self._out("transcript.jsonl"))
        actions = scorer.build_action_stream(events)

        run_score = scorer.score_run(
            self.leg_spec, workspace=self.workspace,
            leg_start_baselines=self.leg_start_baselines,
            transcript=events, verify_dir=self.verify_dir,
        )
        cost = scorer.cost_metrics(events)
        metrics = scorer.leg_metrics(
            self.leg_spec, events=events, actions=actions, workspace=self.workspace,
            leg_start_baselines=self.leg_start_baselines, k=self.k, origin=self.scenario.origin,
            session_usd=cost.get("session_usd"),
            billable_tokens_session=cost.get("billable_tokens_session"),
        )
        variant = self._find_note_variant() if self.note_variant != "none" else None
        metrics.update(
            scorer.t3_metrics(
                variant, fact_sentence=self.scenario.fact_sentence, actions=actions,
                scenario_anchors=self.scenario.anchor_files(),
            )
        )
        run_score["contradictions"] = scorer.detect_contradictions(
            self.leg_spec, checks=run_score["checks"], metrics=metrics
        )

        nv = scorer.leg_non_vacuity(
            arm=self.arm, k=self.k, events=events,
            notes_count_at_start=self.notes_count_at_start,
            restored_manifest_ok=self.restored_manifest_ok,
            audit_log=self.audit_log, audit_since_offset=self.audit_offset,
            proxy_injected=(self.record.get("proxy_metrics") or {}).get("injected"),
            planted_anchor=self.planted_anchor,
            hook_injection_counts=self.record.get("hook_injection_counts"),
            user_prompt_submit_injection_delta=self.record.get("user_prompt_submit_injection_delta"),
            transcript_path=self._out("transcript.jsonl"),
            planted_note_content=(variant.content if variant is not None else None),
            note_id=self.note_id,
            recall_probe_returned_note=self.recall_probe_returned_note,
            mcp_handshake_ok=self.record.get("mcp_handshake_ok"),
            trail_text_delivered=self.trail_text_delivered,
            agent_returncode=self.record.get("agent_returncode"),
            is_error=cost.get("is_error"),
            output_tokens=cost.get("output_tokens"),
            channel_delivery=self.record.get("channel_delivery"),
        )

        self.record["score"] = run_score
        self.record["metrics"] = metrics
        self.record["cost"] = cost
        self.record["valid"] = nv["valid"]
        self.record["invalid_reason"] = nv["invalid_reason"]
        self.record["non_vacuity"] = nv["non_vacuity"]

    def snapshot(self) -> None:
        end_baselines = _sha256_tree(self.workspace, extra_ignore=self._extra_ignore_paths())
        tar_path = self.root / ("workspace.tar" if self.shared_leg1 else "end-state.tar")
        manifest_path = self.root / "manifest.sha256"
        _make_tar(self.workspace, tar_path)
        manifest_hash = _write_manifest(end_baselines, manifest_path)
        self.record["end_state_manifest_sha256"] = manifest_hash
        if self.shared_leg1:
            leg1_id = hashlib.sha256(tar_path.read_bytes()).hexdigest()
            (self.root / "leg1_id").write_text(leg1_id)
            self.record["leg1_id"] = leg1_id
        else:
            self.record["end_state_tar"] = str(tar_path)

    def write(self) -> None:
        self._out("result.json").write_text(json.dumps(self.record, indent=2))
        if not self.shared_leg1:
            runs_dir = self.root.parents[2]
            with (runs_dir / "results.jsonl").open("a") as fh:
                fh.write(json.dumps(self.record) + "\n")

    def report(self) -> None:
        r, s, m, cost = self.record, self.record.get("score", {}), self.record.get("metrics", {}), self.record.get("cost", {})
        print("=" * 72)
        print(f"leg {r['leg_id']}")
        print(f"  scenario : {r['scenario']} ({r['origin']})  trajectory: {r.get('trajectory_id')}")
        print(f"  arm      : {r['arm']}   note_variant: {r['note_variant']}   k={r['k']}   model: {r['model']}")
        print(f"  valid    : {r.get('valid')}" + (f"  -- {r.get('invalid_reason')}" if r.get("invalid_reason") else ""))
        print()
        print(f"  PRIMARY CHECK  {s.get('primary_check')} -> fact_used={s.get('fact_used')}")
        for c in s.get("checks", []):
            print(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:32s} {c['detail']}")
        print()
        print(
            f"  censored={m.get('censored')} mistake_committed={m.get('mistake_committed')} "
            f"turns_to_fact={m.get('turns_to_fact')} tool_calls_to_fact={m.get('tool_calls_to_fact')} "
            f"vectr_tool_calls={m.get('vectr_tool_calls')}"
        )
        print(
            f"  cost: turns={cost.get('session_turns')} tool_calls={cost.get('tool_calls')} "
            f"usd={cost.get('session_usd')} wall={r.get('agent_wall_s')}s"
        )
        print(f"  artifacts: {self.root}")
        print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=sorted(scen.SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--shared-leg1", action="store_true", help=(
        "Run leg 1 under arm-A conditions and write self-contained artifacts to "
        "--out-dir (DESIGN.md 5.2, 8). Mutually exclusive with --k/--arm/--leg-dir/"
        "--workspace-dir/--trajectory-id/--db-dir/--note-variant/--plant-note."
    ))
    ap.add_argument("--out-dir", help="Output dir for --shared-leg1 mode.")

    ap.add_argument("--k", type=int, help="Leg number (>= 1) for normal mode.")
    ap.add_argument("--leg-dir", help="Output dir for a normal leg, e.g. <runs-dir>/<trajectory>/legs/<k>/.")
    ap.add_argument("--workspace-dir", help=(
        "Agent/daemon workspace directory. Pass the SAME path for every leg of one "
        "trajectory (e.g. <runs-dir>/<trajectory>/workspace) so a note a leg plants "
        "is registered under a workspace path a later leg's fresh daemon still "
        "queries against -- vectr's working-memory store scopes notes by workspace "
        "path, not by --db-dir alone. Defaults to <leg-dir>/workspace (per-leg) if "
        "omitted, which is only correct for a single-shot leg that is never chained "
        "with a later k of the same trajectory (e.g. T0's probe cells)."
    ))
    ap.add_argument("--trajectory-id")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--note-variant", choices=NOTE_VARIANTS, default="none")
    ap.add_argument("--plant-note", action="store_true", help=(
        "Plant the canonical note for --note-variant on THIS leg (exactly once per "
        "trajectory, immediately after leg 1, for every memory arm -- DESIGN.md 5.3)."
    ))
    ap.add_argument("--planted-note-id", type=int, default=None, help=(
        "Pass-through note id for a leg that did not itself plant. Ignored if --plant-note is given."
    ))
    ap.add_argument("--planted-anchor", default=None, help=(
        "Pass-through 'note:<id>' anchor string, same rules as --planted-note-id."
    ))
    ap.add_argument("--db-dir", help="VECTR_DB_DIR to create (if new) or reuse across this trajectory's legs.")
    ap.add_argument("--restore-tar", help="Snapshot tar to restore into workspace/ before this leg (required for k > 1).")
    ap.add_argument("--restore-manifest-sha256", help="Expected manifest sha256 for --restore-tar; mismatch aborts.")
    ap.add_argument("--allow-manifest-mismatch", action="store_true", help=(
        "Continue past a restored-snapshot manifest mismatch instead of aborting (debugging only)."
    ))
    ap.add_argument("--hook-attestation", help=(
        "Path to a JSON {verified,date,method,claude_code_version} file; required for --arm hook-full."
    ))

    ap.add_argument("--daemon-port", type=int, default=DEFAULT_DAEMON_PORT)
    ap.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--vectr-bin", default=shutil.which("vectr") or "vectr")
    ap.add_argument("--allow-unreachable", action="store_true", help=(
        "Record a leg even on GENUINE preflight unreachability -- the planted note is "
        "absent/content-corrupt in the store (by-id /v1/recall check), the SessionStart "
        "channel never carries it (hook arms), or every probe path fails at the "
        "transport level. Does NOT apply to mere channel displacement (a higher-ranked "
        "note winning the proactive channel's default slot): that is recorded via "
        "channel_delivery='displaced' and the leg runs regardless "
        "(UPG-EVAL-PLANT-DISPLACEMENT)."
    ))
    ap.add_argument("--allow-hook-unreachable", action="store_true", help=(
        "Record a leg even when hook_preflight() cannot prove the SessionStart "
        "hook mechanism delivers the planted note on this leg's own scratch "
        "daemon (arms hook-sessionstart/hook-full only)."
    ))
    ap.add_argument("--allow-store-mismatch", action="store_true", help=(
        "Record a leg even when verify_note_store_matched() finds "
        "notes_in_store_at_start != 1 for an arm with no live vectr tool access "
        "(UPG-EVAL-STORE-MATCH). Debugging only -- a real campaign run should "
        "never need this; it means the trajectory's --db-dir carries stray notes."
    ))
    ap.add_argument("--probe-only", action="store_true", help=(
        "ZERO-QUOTA: materialize/restore, start the daemon, optionally plant, run "
        "the daemon-side probes, tear down, print a summary. Spawns no agent, writes "
        "no result.json/snapshot. With k > 1 and no --restore-tar, materializes "
        "fresh instead of requiring a real prior-leg snapshot -- the scenario/note "
        "reachability smoke check T0 uses (DESIGN.md 9)."
    ))
    return ap


def _validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.shared_leg1:
        if not args.out_dir:
            ap.error("--out-dir is required with --shared-leg1")
        for flag, val in (
            ("--k", args.k), ("--arm", args.arm), ("--leg-dir", args.leg_dir),
            ("--workspace-dir", args.workspace_dir),
            ("--trajectory-id", args.trajectory_id), ("--db-dir", args.db_dir),
            ("--restore-tar", args.restore_tar), ("--planted-note-id", args.planted_note_id),
            ("--planted-anchor", args.planted_anchor),
        ):
            if val is not None:
                ap.error(f"{flag} is not accepted with --shared-leg1")
        if args.note_variant != "none":
            ap.error("--note-variant is not accepted with --shared-leg1 (leg 1 never plants a note)")
        if args.plant_note:
            ap.error("--plant-note is not accepted with --shared-leg1 (leg 1 never plants a note)")
        return

    if args.k is None:
        ap.error("--k is required unless --shared-leg1")
    if args.k < 1:
        ap.error("--k must be >= 1")
    if not args.leg_dir:
        ap.error("--leg-dir is required unless --shared-leg1")
    if not args.arm:
        ap.error("--arm is required unless --shared-leg1")
    if not args.trajectory_id:
        ap.error("--trajectory-id is required unless --shared-leg1")
    if not args.db_dir:
        ap.error("--db-dir is required unless --shared-leg1")
    if args.k > 1 and not args.restore_tar and not args.probe_only:
        ap.error("--restore-tar is required for k > 1 (unless --probe-only)")
    if args.restore_tar and not args.restore_manifest_sha256:
        ap.error("--restore-manifest-sha256 is required together with --restore-tar")
    if args.plant_note and args.arm == "none":
        ap.error("--plant-note is invalid for arm 'none' (arm A never has memory)")
    if args.plant_note and args.note_variant == "none":
        ap.error("--plant-note requires --note-variant plain|provenance|verifiable")
    if args.arm == "proxy" and args.k > 1 and not args.plant_note and not args.planted_anchor and not args.probe_only:
        ap.error(
            "--planted-anchor is required for arm 'proxy' at k > 1 when this leg "
            "does not itself plant the note (needed to match PROACTIVE_INJECT lines)"
        )


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()
    _validate_args(ap, args)

    if args.daemon_port < 8899 or args.proxy_port < 8899:
        raise SystemExit("ABORT: scratch ports must be >= 8899 (8765/8800 are live daemons)")

    hook_attestation = _enforce_hook_attestation(
        args.arm if not args.shared_leg1 else "none", args.hook_attestation
    )

    runner = LegRunner(args)
    if hook_attestation is not None:
        runner.record["hook_attestation"] = hook_attestation

    if args.probe_only:
        try:
            runner.prepare()
            runner.start_daemon()
            if args.plant_note:
                runner.plant_note()
            runner.verify_note_store_matched()
            runner.probe()
        finally:
            _stop_daemon(args.vectr_bin, runner.workspace, runner.daemon_port, runner._daemon_env())
        print(
            f"probe-only {runner.scenario.slug} k={runner.k} arm={runner.arm} "
            f"note_variant={runner.note_variant}"
        )
        print(f"  planted_anchor       : {runner.planted_anchor}")
        print(f"  reachable            : {runner.record.get('planted_note_reachable_preflight')}")
        print(f"  trail_text_delivered : {runner.record.get('trail_text_delivered')}")
        print(f"  recall_probe         : {runner.record.get('recall_probe_returned_note')}")
        return

    try:
        runner.prepare()
        runner.start_daemon()
        if args.plant_note:
            runner.plant_note()
        runner.verify_note_store_matched()
        runner.probe()
        runner.start_proxy()
        if runner.arm in ("hook-sessionstart", "hook-full", "hook-userpromptsubmit"):
            runner.write_hooks()
        if runner.arm in ("hook-sessionstart", "hook-full"):
            runner.hook_preflight()
        runner.run_agent()
    finally:
        try:
            runner.capture_and_teardown()
        except Exception as exc:  # teardown must never mask a run
            print(f"[run_leg] teardown issue: {type(exc).__name__}: {exc}", file=sys.stderr)
            _hard_stop_port(runner.proxy_port)
            _hard_stop_port(runner.daemon_port)

    runner.score()
    runner.snapshot()
    runner.write()
    runner.report()

    _abort_if_session_errored(runner.record)


def _abort_if_session_errored(record: dict[str, Any]) -> None:
    """A genuine abort, not an ordinary invalid leg (module docstring's exit-code
    contract): if `scorer.leg_non_vacuity`'s arm-agnostic `session_errored` gate
    fired, the agent session itself errored or produced no output, so this leg's
    result.json/snapshot describe a session that never ran meaningfully. A nonzero
    exit here is what tells a driver (run_plan.py) to refuse to cache/chain this
    leg's end state (`run_plan.py`'s `_ensure_shared_leg1` and its `rc != 0`
    handling in the k-loop). Kept standalone (not inlined in `main()`) so it can be
    exercised directly against a constructed record without spawning a real agent
    session.
    """
    if (record.get("non_vacuity") or {}).get("session_errored"):
        raise SystemExit(f"ABORT: {record.get('invalid_reason')}")


if __name__ == "__main__":
    main()
