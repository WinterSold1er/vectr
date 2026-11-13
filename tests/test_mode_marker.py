"""UPG-MODE-MARKER: `<workspace>/.vectr/mode` outlives the registry record.

Why this file exists. `agent/instance_registry.py`'s `prune_dead()` deletes every
dead entry wholesale — `mode` field included — and it sweeps the *whole* registry
in one pass. So a host restart that kills every daemon (the incident: a
`systemctl --user restart` with `KillMode=control-group` SIGTERM'd 25 of them)
erases the recorded mode of all 25 workspaces at once: only the first `start`
afterwards can still read a record, and every later one falls back to full mode.
That is exactly how `/home/csy` — recorded `memory-only` — was relaunched in full
mode by the DSH plugin's flag-less `vectr start` and spent five CPU-hours
rebuilding a 1.95M-chunk index.

`<workspace>/.vectr/mode` (plain text, one of `full` / `memory-only` /
`search-only`) is the per-workspace copy that survives the sweep. Precedence is
explicit flag > registry record > marker > full, and the marker is rewritten on
every successful launch — `full` included, so an explicit `--full` clears it.

No daemon is launched anywhere in this file. The `cmd_*` tests mock
`main._do_start`; the single `_do_start` test fakes `subprocess.Popen` and the
readiness probe. The registry is always a tmp_path file, never
`~/.vectr/instances.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.instance_registry import InstanceRegistry, workspace_hash
import main as m


def _ws(tmp_path) -> str:
    """A real workspace directory under tmp_path (the marker needs a real dir)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    return str(ws)


def _marker(ws: str) -> Path:
    """The documented on-disk location, spelled out rather than via main's
    helper — so a move of the marker file shows up as a test failure here."""
    return Path(ws) / ".vectr" / "mode"


def _seed_marker(ws: str, mode: str) -> None:
    """What an earlier successful launch left behind, written raw: the
    read-side tests must not depend on the writer being correct."""
    path = _marker(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mode + "\n", encoding="utf-8")


def _registry(tmp_path) -> InstanceRegistry:
    return InstanceRegistry(registry_path=tmp_path / "instances.json")


def _start_args(ws: str, **kwargs) -> argparse.Namespace:
    """`vectr start`'s parsed args, with every mode flag named explicitly."""
    defaults = {
        "workspace": None,
        "paths": [ws],
        "port": 8765,
        "memory_only": False,
        "search_only": False,
        "full": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _run_start(reg: InstanceRegistry, args: argparse.Namespace, *, alive: bool = False):
    """Run `cmd_start` against a tmp registry with the daemon launch mocked out."""
    with patch("main.InstanceRegistry", return_value=reg), \
         patch("agent.instance_registry._is_pid_alive", return_value=alive), \
         patch("main._is_pid_alive", return_value=alive), \
         patch("agent.instance_registry._port_is_free", return_value=True), \
         patch("main._write_workspace_config"), \
         patch("main._preflight_grammars"), \
         patch("main._do_start") as mock_do_start:
        m.cmd_start(args)
    return mock_do_start


def _run_restart(reg: InstanceRegistry, args: argparse.Namespace):
    with patch("main.InstanceRegistry", return_value=reg), \
         patch("agent.instance_registry._port_is_free", return_value=True), \
         patch("agent.instance_registry._is_pid_alive", return_value=False), \
         patch("main._is_pid_alive", return_value=False), \
         patch("main._stop_server"), \
         patch("main._write_workspace_config"), \
         patch("main._do_start") as mock_do_start:
        m.cmd_restart(args)
    return mock_do_start


# ---------------------------------------------------------------------------
# (a) marker write / read round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["full", "memory-only", "search-only"])
def test_marker_round_trip(tmp_path, mode):
    ws = _ws(tmp_path)

    m._write_mode_marker(ws, mode)

    assert _marker(ws).read_text(encoding="utf-8").strip() == mode
    assert m._read_mode_marker(ws) == mode


def test_missing_marker_reads_as_none(tmp_path):
    """A workspace created before this marker existed: nothing to inherit."""
    assert m._read_mode_marker(_ws(tmp_path)) is None


def test_marker_lives_at_the_documented_path(tmp_path):
    """`<workspace>/.vectr/mode` — next to `.vectr/style`, and never *in* it:
    `style` carries the instruction style (`memory-first` etc.) and does not
    control indexing."""
    ws = _ws(tmp_path)

    assert m._mode_marker_path(ws) == Path(ws) / ".vectr" / "mode"
    assert m._mode_marker_path(ws).name == "mode"
    m._write_mode_marker(ws, "memory-only")
    assert _marker(ws).read_text(encoding="utf-8").strip() == "memory-only"
    assert not (Path(ws) / ".vectr" / "style").exists()


@pytest.mark.parametrize(
    "garbage",
    ["", "   \n", "MEMORY-ONLY", "memory_only", "memory-first", "full\nmemory-only",
     "{\"mode\": \"memory-only\"}", "nonsense"],
)
def test_garbage_marker_is_ignored_without_crashing(tmp_path, garbage):
    """Unrecognised content must read as "no marker" (→ full), never raise and
    never be half-trusted."""
    ws = _ws(tmp_path)
    _marker(ws).parent.mkdir(parents=True, exist_ok=True)
    _marker(ws).write_text(garbage, encoding="utf-8")

    assert m._read_mode_marker(ws) is None


def test_unreadable_marker_reads_as_none(tmp_path):
    """A marker that is a directory (or otherwise unreadable) is None, not an
    exception escaping into the launch path."""
    ws = _ws(tmp_path)
    _marker(ws).parent.mkdir(parents=True, exist_ok=True)
    _marker(ws).mkdir()

    assert m._read_mode_marker(ws) is None


def test_write_normalises_unknown_mode_to_full(tmp_path):
    ws = _ws(tmp_path)

    m._write_mode_marker(ws, "definitely-not-a-mode")

    assert m._read_mode_marker(ws) == "full"


def test_write_replaces_a_stale_marker(tmp_path):
    """`--full` has to be able to correct a `memory-only` marker."""
    ws = _ws(tmp_path)
    m._write_mode_marker(ws, "memory-only")

    m._write_mode_marker(ws, "full")

    assert m._read_mode_marker(ws) == "full"


# ---------------------------------------------------------------------------
# (e) the write is atomic
# ---------------------------------------------------------------------------

def test_write_is_atomic_and_leaves_no_tmp_file(tmp_path):
    """Content lands via a scratch file + `os.replace`, never in place, and no
    scratch file is left behind for the next reader to trip over.

    The scratch name is unique per write and lives in the marker's own directory:
    a fixed `mode.tmp` is shared by racing writers (see the collision test in
    section (g)), and a scratch file anywhere off that directory makes the
    `os.replace` a cross-filesystem copy instead of an atomic rename."""
    ws = _ws(tmp_path)
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src, dst, *a, **kw):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst, *a, **kw)

    with patch("main.os.replace", side_effect=spy):
        m._write_mode_marker(ws, "memory-only")

    assert len(replaced) == 1, "the marker must be published with one os.replace"
    src, dst = replaced[0]
    assert dst == str(_marker(ws))
    scratch = Path(src)
    assert str(scratch.parent) == str(_marker(ws).parent), "scratch file must be a sibling of the marker"
    assert scratch.name.startswith(".mode.") and scratch.name.endswith(".tmp")
    assert scratch.name != "mode.tmp", "the scratch name must be unique, not a fixed shared name"
    assert not scratch.exists(), "the tmp file must not survive the replace"
    assert [p.name for p in _marker(ws).parent.iterdir()] == ["mode"]
    assert m._read_mode_marker(ws) == "memory-only"


def test_write_failure_is_announced_and_not_fatal(tmp_path, capsys):
    """A read-only workspace must still be able to run a daemon: the failure is
    reported and the launch continues (no exception, no marker)."""
    ws = _ws(tmp_path)
    _marker(ws).parent.mkdir(parents=True, exist_ok=True)
    os.chmod(_marker(ws).parent, 0o500)
    try:
        m._write_mode_marker(ws, "memory-only")
    finally:
        os.chmod(_marker(ws).parent, 0o700)

    err = capsys.readouterr().err
    assert "Warning: could not record memory-only mode" in err
    assert m._read_mode_marker(ws) is None


# ---------------------------------------------------------------------------
# (b) the regression: start after prune_dead() wiped the record
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode, expect_memory, expect_search",
    [("memory-only", True, False), ("search-only", False, True)],
)
def test_start_after_prune_uses_marker(tmp_path, mode, expect_memory, expect_search):
    """The 25-daemon fleet case: an earlier `start` already swept *every* dead
    entry out of the registry, so this workspace has no record at all. The
    marker is the only surviving copy of its mode, and a flag-less `start` (what
    the plugin's startupHeal sends) must reproduce it."""
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)
    wh = workspace_hash(ws)
    reg.register(wh, ws, 8765, 99999, mode=mode)
    _seed_marker(ws, mode)  # what the original launch recorded
    # ... the host restart kills the daemon, then the FIRST other workspace's
    # start sweeps the whole registry, this entry included:
    reg.prune_dead()
    import json
    assert json.loads((tmp_path / "instances.json").read_text()) == {}
    assert reg.get(wh) is None

    mock_do_start = _run_start(reg, _start_args(ws))

    assert mock_do_start.call_args.kwargs["memory_only"] is expect_memory
    assert mock_do_start.call_args.kwargs["search_only"] is expect_search


def test_start_with_only_a_marker_works_without_any_registry_file(tmp_path):
    """No instances.json at all (fresh-ish host, registry deleted/recreated):
    the marker alone decides."""
    ws = _ws(tmp_path)
    _seed_marker(ws, "memory-only")
    reg = _registry(tmp_path)
    assert not (tmp_path / "instances.json").exists()

    mock_do_start = _run_start(reg, _start_args(ws))

    assert mock_do_start.call_args.kwargs["memory_only"] is True


def test_start_without_marker_or_record_is_full(tmp_path):
    """Backwards compatibility: a workspace with neither record nor marker
    behaves exactly as before this change."""
    ws = _ws(tmp_path)

    mock_do_start = _run_start(_registry(tmp_path), _start_args(ws))

    assert mock_do_start.call_args.kwargs["memory_only"] is False
    assert mock_do_start.call_args.kwargs["search_only"] is False


# ---------------------------------------------------------------------------
# (c) an explicit flag always wins over the marker
# ---------------------------------------------------------------------------

def test_explicit_full_overrides_marker(tmp_path):
    ws = _ws(tmp_path)
    _seed_marker(ws, "memory-only")

    mock_do_start = _run_start(_registry(tmp_path), _start_args(ws, full=True))

    assert mock_do_start.call_args.kwargs["memory_only"] is False
    assert mock_do_start.call_args.kwargs["search_only"] is False


def test_explicit_memory_only_overrides_full_marker(tmp_path):
    ws = _ws(tmp_path)
    _seed_marker(ws, "full")

    mock_do_start = _run_start(_registry(tmp_path), _start_args(ws, memory_only=True))

    assert mock_do_start.call_args.kwargs["memory_only"] is True


def test_registry_record_beats_marker(tmp_path):
    """When both exist they are written by the same launch and agree — but if
    they ever disagree, the live record (written after the marker) wins."""
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)
    reg.register(workspace_hash(ws), ws, 8765, 99999, mode="full")
    _seed_marker(ws, "memory-only")

    mock_do_start = _run_start(reg, _start_args(ws))

    assert mock_do_start.call_args.kwargs["memory_only"] is False


# ---------------------------------------------------------------------------
# (d) restart falls back to the marker when there is no record
# ---------------------------------------------------------------------------

def test_restart_falls_back_to_marker(tmp_path):
    ws = _ws(tmp_path)
    _seed_marker(ws, "memory-only")
    reg = _registry(tmp_path)

    with patch("main._resolve_restart_roots", return_value=([ws], None)):
        mock_do_start = _run_restart(reg, _start_args(ws))

    assert mock_do_start.call_args.kwargs["memory_only"] is True


def test_restart_explicit_full_still_wins_over_marker(tmp_path):
    ws = _ws(tmp_path)
    _seed_marker(ws, "memory-only")
    reg = _registry(tmp_path)

    with patch("main._resolve_restart_roots", return_value=([ws], None)):
        mock_do_start = _run_restart(reg, _start_args(ws, full=True))

    assert mock_do_start.call_args.kwargs["memory_only"] is False


# ---------------------------------------------------------------------------
# The lifecycle around the marker: written on launch, kept by `stop`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mode_kwargs, expected",
    [({"memory_only": True}, "memory-only"),
     ({"search_only": True}, "search-only"),
     ({}, "full")],
)
def test_successful_launch_writes_marker(tmp_path, mode_kwargs, expected):
    """The single write point is `_do_start`, i.e. a launch that actually
    spawned a daemon — and `full` included, so an explicit `--full` corrects a
    stale marker instead of being dragged back by it."""
    ws = _ws(tmp_path)
    # Seed the *opposite* mode, so each case also proves the launch overwrites a
    # stale marker rather than merely writing where none existed.
    _seed_marker(ws, "memory-only" if expected == "full" else "full")
    reg = _registry(tmp_path)

    class _FakeProc:
        pid = 424242

    import pathlib

    with patch.dict(os.environ, {}, clear=False), \
         patch("main.subprocess.Popen", return_value=_FakeProc()) as popen, \
         patch("main.InstanceRegistry", return_value=reg), \
         patch("main._migrate_legacy_files"), \
         patch("main._wait_for_daemon_ready", return_value=True), \
         patch("main._is_pid_alive", return_value=True), \
         patch("agent.fs_permissions.secure_dir"), \
         patch("pathlib.Path.home", return_value=tmp_path):
        os.environ.pop("VECTR_MEMORY_ONLY", None)
        os.environ.pop("VECTR_SEARCH_ONLY", None)
        m._do_start(ws, 8765, workspace_hash(ws), **mode_kwargs)

    popen.assert_called_once()
    assert m._read_mode_marker(ws) == expected
    # ... and it agrees with the registry record written a line earlier.
    assert reg.get(workspace_hash(ws))["mode"] == expected


def test_stop_keeps_the_marker_and_a_later_start_reproduces_it(tmp_path):
    """`vectr stop` unregisters the workspace; the marker must survive it, so a
    workspace that was deliberately run memory-only does not silently come back
    as a full-mode indexer after a stop/start cycle."""
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)
    wh = workspace_hash(ws)
    reg.register(wh, ws, 8765, 12345, mode="memory-only")
    _seed_marker(ws, "memory-only")

    reg.unregister(wh)  # what `vectr stop` does

    assert reg.get(wh) is None
    assert m._read_mode_marker(ws) == "memory-only"
    mock_do_start = _run_start(reg, _start_args(ws))
    assert mock_do_start.call_args.kwargs["memory_only"] is True


def test_already_running_start_backfills_the_marker(tmp_path):
    """A daemon started by a vectr older than this marker is live but has no
    durable mode. `start` (the plugin's startupHeal sends one on every reload)
    must record the live entry's own mode while it is still readable, so the
    next kill/restart does not lose it."""
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)
    reg.register(workspace_hash(ws), ws, 8765, 12345, mode="memory-only")
    assert m._read_mode_marker(ws) is None

    mock_do_start = _run_start(reg, _start_args(ws), alive=True)

    mock_do_start.assert_not_called()
    assert m._read_mode_marker(ws) == "memory-only"


# ---------------------------------------------------------------------------
# (g) the hardened read/write paths (adversarial review D1-D6)
# ---------------------------------------------------------------------------

def _seed_raw_marker(ws: str, payload: bytes) -> None:
    """A marker as it exists in the wild after something else wrote the file:
    raw bytes, no encoding guarantee."""
    path = _marker(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _strip_mode_key(registry_file: Path, ws_hash: str) -> None:
    """Rewrite a recorded entry the way a vectr from before the `mode` field wrote
    it: every other field present, `mode` absent."""
    data = json.loads(registry_file.read_text())
    del data[ws_hash]["mode"]
    registry_file.write_text(json.dumps(data))


def _launch_mocked(tmp_path, ws: str, reg: InstanceRegistry, *, ready: bool, alive: bool,
                   **mode_kwargs) -> None:
    """`_do_start` with a fake daemon process: `ready`/`alive` pick which of the
    three launch outcomes (ready / not_ready / failed) is being exercised."""

    class _FakeProc:
        pid = 424242

    with patch.dict(os.environ, {}, clear=False), \
         patch("main.subprocess.Popen", return_value=_FakeProc()), \
         patch("main.InstanceRegistry", return_value=reg), \
         patch("main._migrate_legacy_files"), \
         patch("main._wait_for_daemon_ready", return_value=ready), \
         patch("main._is_pid_alive", return_value=alive), \
         patch("agent.fs_permissions.secure_dir"), \
         patch("pathlib.Path.home", return_value=tmp_path):
        os.environ.pop("VECTR_MEMORY_ONLY", None)
        os.environ.pop("VECTR_SEARCH_ONLY", None)
        m._do_start(ws, 8765, workspace_hash(ws), **mode_kwargs)


def test_non_utf8_marker_reads_as_none_with_a_warning(tmp_path, capsys):
    """D1: `UnicodeDecodeError` descends from ValueError, so an `except OSError`
    read swallows permissions but NOT a binary marker — and the traceback landed
    in `cmd_start`. It must degrade to 'no usable mode', announced."""
    ws = _ws(tmp_path)
    _seed_raw_marker(ws, b"\x00\xff\xfe")

    assert m._read_mode_marker(ws) is None

    err = capsys.readouterr().err
    assert str(_marker(ws)) in err, "the warning must name the marker it could not read"
    assert "fall back to full mode" in err


def test_start_survives_a_binary_marker_and_launches_full(tmp_path, capsys):
    """The reviewer's reproduction: `vectr start` on a workspace whose marker is
    raw bytes crashed instead of launching. It now launches in full mode, loudly."""
    ws = _ws(tmp_path)
    _seed_raw_marker(ws, b"\x00\xff\xfe")

    mock_do_start = _run_start(_registry(tmp_path), _start_args(ws))

    mock_do_start.assert_called_once()
    assert mock_do_start.call_args.kwargs["memory_only"] is False
    assert mock_do_start.call_args.kwargs["search_only"] is False
    err = capsys.readouterr().err
    assert "could not be used" in err
    assert str(_marker(ws)) in err


def test_unreadable_marker_is_announced_and_falls_back_to_full(tmp_path, capsys, monkeypatch):
    """D2: a marker that is there but cannot be read (`chmod 000`) used to produce
    zero diagnostics and a silent full-mode launch — the original incident wearing
    a different hat."""
    monkeypatch.delenv("VECTR_MEMORY_ONLY", raising=False)
    monkeypatch.delenv("VECTR_SEARCH_ONLY", raising=False)
    ws = _ws(tmp_path)
    _seed_marker(ws, "memory-only")
    os.chmod(_marker(ws), 0o000)
    try:
        mock_do_start = _run_start(_registry(tmp_path), _start_args(ws))
    finally:
        os.chmod(_marker(ws), 0o600)

    err = capsys.readouterr().err
    assert "could not be used" in err, "an unusable marker must never fall back to full in silence"
    assert "full mode" in err
    assert str(_marker(ws)) in err
    assert mock_do_start.call_args.kwargs["memory_only"] is False


def test_missing_marker_stays_silent(tmp_path, capsys, monkeypatch):
    """The other half of D2: 'no marker' is the documented default and must not be
    noisy, or the warning for a real problem stops meaning anything."""
    monkeypatch.delenv("VECTR_MEMORY_ONLY", raising=False)
    monkeypatch.delenv("VECTR_SEARCH_ONLY", raising=False)
    ws = _ws(tmp_path)

    mock_do_start = _run_start(_registry(tmp_path), _start_args(ws))

    mock_do_start.assert_called_once()
    assert capsys.readouterr().err == ""


def test_concurrent_marker_writers_do_not_share_scratch(tmp_path):
    """D3: two writers that both create their scratch file before either publishes
    it. With one fixed `mode.tmp` the loser's `os.replace` dies with [Errno 2] and
    the mode it was asked to record is never recorded; unique scratch names make
    both publishes succeed."""
    ws = _ws(tmp_path)
    barrier = threading.Barrier(2, timeout=10)
    outcomes: list[tuple[str, BaseException | None]] = []
    real_replace = os.replace

    def spy(src, dst, *a, **kw):
        barrier.wait()  # both writers have created their scratch file before either publishes
        try:
            result = real_replace(src, dst, *a, **kw)
        except BaseException as exc:  # recorded, then re-raised: the writer must see it
            outcomes.append((str(src), exc))
            raise
        outcomes.append((str(src), None))
        return result

    errors: list[BaseException] = []

    def writer(mode: str) -> None:
        try:
            m._write_mode_marker(ws, mode)
        except BaseException as exc:  # noqa: BLE001 - reported below, never swallowed
            errors.append(exc)

    with patch("main.os.replace", side_effect=spy):
        threads = [threading.Thread(target=writer, args=(mode,)) for mode in ("memory-only", "search-only")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert not errors, f"a racing writer must not fail: {errors}"
    assert [exc for _, exc in outcomes] == [None, None], f"both publishes must land: {outcomes}"
    assert len({src for src, _ in outcomes}) == 2, "each writer needs its own scratch file"
    assert m._read_mode_marker(ws) in ("memory-only", "search-only")
    assert [p.name for p in _marker(ws).parent.iterdir()] == ["mode"]


def test_failed_launch_does_not_create_a_marker(tmp_path):
    """D6: exit_code=1 / daemon never ready used to leave a marker behind, i.e. a
    launch that served nothing became the workspace's remembered mode."""
    ws = _ws(tmp_path)

    with pytest.raises(SystemExit) as exc:
        _launch_mocked(tmp_path, ws, _registry(tmp_path), ready=False, alive=False, memory_only=True)

    assert exc.value.code == 1
    assert not _marker(ws).exists(), "a daemon that never came up records nothing"


def test_failed_launch_does_not_rewrite_an_existing_marker(tmp_path):
    ws = _ws(tmp_path)
    _seed_marker(ws, "memory-only")

    with pytest.raises(SystemExit) as exc:
        _launch_mocked(tmp_path, ws, _registry(tmp_path), ready=False, alive=False)

    assert exc.value.code == 1
    assert m._read_mode_marker(ws) == "memory-only"


def test_not_ready_launch_leaves_the_marker_alone(tmp_path):
    """`not_ready` is not `failed`, but it is not proof the daemon serves anything
    either: the rule is one write point, after `_wait_for_daemon_ready` succeeds."""
    ws = _ws(tmp_path)
    _seed_marker(ws, "search-only")

    _launch_mocked(tmp_path, ws, _registry(tmp_path), ready=False, alive=True)

    assert m._read_mode_marker(ws) == "search-only"


def test_already_running_without_recorded_mode_keeps_the_marker(tmp_path):
    """D4: `entry.get("mode") or _mode_name(...)` fell back to the flags of a call
    that started no daemon. For the plain `vectr start` an IDE plugin sends on
    every reload, that rewrote a deliberate memory-only marker to full."""
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)
    wh = workspace_hash(ws)
    reg.register(wh, ws, 8765, 12345, mode="memory-only")
    _strip_mode_key(tmp_path / "instances.json", wh)
    _seed_marker(ws, "memory-only")

    mock_do_start = _run_start(reg, _start_args(ws), alive=True)

    mock_do_start.assert_not_called()
    assert m._read_mode_marker(ws) == "memory-only"
    # ... and the record it declined to read a mode from is still mode-less.
    assert "mode" not in json.loads((tmp_path / "instances.json").read_text())[wh]


def test_already_running_without_recorded_mode_invents_nothing(tmp_path):
    """Same record, no marker: the answer is still no marker. An absent mode in the
    record is not evidence for `full`."""
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)
    wh = workspace_hash(ws)
    reg.register(wh, ws, 8765, 12345, mode="memory-only")
    _strip_mode_key(tmp_path / "instances.json", wh)
    assert m._read_mode_marker(ws) is None

    _run_start(reg, _start_args(ws), alive=True)

    assert not _marker(ws).exists()


def test_full_flag_conflicts_with_memory_only_env_var(tmp_path, capsys, monkeypatch):
    """D5: `VECTR_MEMORY_ONLY=1` used to override an explicit `--full` in silence
    AND persist that verdict. Explicit `--full` plus an env var demanding
    memory-only is a contradiction, so it gets the flag/flag verdict: error, exit 1."""
    monkeypatch.setenv("VECTR_MEMORY_ONLY", "1")
    monkeypatch.delenv("VECTR_SEARCH_ONLY", raising=False)
    ws = _ws(tmp_path)
    _seed_marker(ws, "memory-only")

    with pytest.raises(SystemExit) as exc:
        _run_start(_registry(tmp_path), _start_args(ws, full=True))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "--full cannot be combined" in err
    assert "VECTR_MEMORY_ONLY=1 requests memory-only mode" in err, "the message must name the variable to unset"
    assert m._read_mode_marker(ws) == "memory-only", "a refused launch must not rewrite the marker"


def test_restart_full_flag_conflicts_with_memory_only_env_var(tmp_path, capsys, monkeypatch):
    """`restart` shares the guard; the incident path runs through it too."""
    monkeypatch.setenv("VECTR_MEMORY_ONLY", "1")
    monkeypatch.delenv("VECTR_SEARCH_ONLY", raising=False)
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)
    reg.register(workspace_hash(ws), ws, 8765, 12345, mode="full")

    with pytest.raises(SystemExit) as exc:
        _run_restart(reg, _start_args(ws, full=True))

    assert exc.value.code == 1
    assert "VECTR_MEMORY_ONLY=1 requests memory-only mode" in capsys.readouterr().err


def test_memory_only_flag_conflicts_with_search_only_env_var(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("VECTR_SEARCH_ONLY", "1")
    monkeypatch.delenv("VECTR_MEMORY_ONLY", raising=False)

    with pytest.raises(SystemExit) as exc:
        _run_start(_registry(tmp_path), _start_args(_ws(tmp_path), memory_only=True))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "memory-only and search-only mode cannot both be requested" in err
    assert "VECTR_SEARCH_ONLY=1 requests search-only mode" in err


def test_both_env_vars_set_is_an_error_not_a_traceback(tmp_path, capsys, monkeypatch):
    """`_effective_mode_flags` raises ValueError on both-set, which reached the CLI
    as an unhandled exception. The guard now turns it into the normal exit-1 error."""
    monkeypatch.setenv("VECTR_MEMORY_ONLY", "1")
    monkeypatch.setenv("VECTR_SEARCH_ONLY", "1")

    with pytest.raises(SystemExit) as exc:
        _run_start(_registry(tmp_path), _start_args(_ws(tmp_path)))

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "memory-only and search-only mode cannot both be requested" in err
    assert "VECTR_MEMORY_ONLY=1 requests memory-only mode" in err
    assert "VECTR_SEARCH_ONLY=1 requests search-only mode" in err


def test_env_var_without_a_flag_still_selects_memory_only(tmp_path, monkeypatch):
    """The guard must not neuter the env var: with no flag typed, VECTR_MEMORY_ONLY=1
    still means memory-only — and a ready daemon still records it. `_do_start` is
    where the env is folded into the mode, so this goes through it directly."""
    monkeypatch.setenv("VECTR_MEMORY_ONLY", "1")
    monkeypatch.delenv("VECTR_SEARCH_ONLY", raising=False)
    ws = _ws(tmp_path)
    reg = _registry(tmp_path)

    class _FakeProc:
        pid = 424242

    with patch("main.subprocess.Popen", return_value=_FakeProc()), \
         patch("main.InstanceRegistry", return_value=reg), \
         patch("main._migrate_legacy_files"), \
         patch("main._wait_for_daemon_ready", return_value=True), \
         patch("main._is_pid_alive", return_value=True), \
         patch("agent.fs_permissions.secure_dir"), \
         patch("pathlib.Path.home", return_value=tmp_path):
        m._do_start(ws, 8765, workspace_hash(ws))  # no mode flag typed

    assert m._read_mode_marker(ws) == "memory-only"
    assert reg.get(workspace_hash(ws))["mode"] == "memory-only"
