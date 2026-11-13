"""UPG-START-DROPS-RECORDED-MODE: `vectr start` reproduces the recorded mode.

`memory-only` / `search-only` live as a runtime field in ~/.vectr/instances.json.
`cmd_start` used to ignore it entirely, so any `start` after the daemon died —
a plugin reload's start-all sweep, a host reboot, a crashed daemon — silently
upgraded a memory-only workspace into a full-mode indexer (the 12 GB index this
regression test exists to prevent). `cmd_restart` already inherited; `cmd_start`
now follows the same precedence, and `--full` is the way back out of an inherited
mode on both commands.

No daemon is launched anywhere in this file: `main._do_start` is mocked, and the
registry is always a tmp_path file, never ~/.vectr/instances.json.
"""
from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import pytest

from agent.instance_registry import InstanceRegistry, workspace_hash
import main as m

WS = "/project/a"


def _registry(tmp_path) -> InstanceRegistry:
    return InstanceRegistry(registry_path=tmp_path / "instances.json")


def _start_args(**kwargs) -> argparse.Namespace:
    """`vectr start`'s parsed args, with every mode flag named explicitly."""
    defaults = {
        "workspace": None,
        "paths": [WS],
        "port": 8765,
        "memory_only": False,
        "search_only": False,
        "full": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _run_start(reg: InstanceRegistry, args: argparse.Namespace, *, alive: bool = False):
    """Run `cmd_start` against a tmp registry with the daemon launch mocked out.

    `_preflight_grammars` is mocked too: it pip-installs missing tree-sitter
    grammars, which is a side effect the test environment must not perform (and
    which a memory-only launch must skip — asserted directly below).
    """
    with patch("main.InstanceRegistry", return_value=reg), \
         patch("agent.instance_registry._is_pid_alive", return_value=alive), \
         patch("main._is_pid_alive", return_value=alive), \
         patch("agent.instance_registry._port_is_free", return_value=True), \
         patch("main._write_workspace_config"), \
         patch("main._preflight_grammars") as mock_preflight, \
         patch("main._do_start") as mock_do_start:
        m.cmd_start(args)
    return mock_do_start, mock_preflight


# ---------------------------------------------------------------------------
# The defect: a dead recorded instance must not come back as full
# ---------------------------------------------------------------------------

def test_dead_recorded_memory_only_starts_memory_only(tmp_path):
    """The exact regression: entry recorded memory-only, pid dead, no flag →
    the daemon must be launched memory-only, not full."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="memory-only")

    mock_do_start, mock_preflight = _run_start(reg, _start_args())

    assert mock_do_start.call_args.kwargs["memory_only"] is True
    assert mock_do_start.call_args.kwargs["search_only"] is False
    # Inheriting memory-only must also skip the grammar preflight a full-mode
    # start performs — the mode is resolved before those downstream gates.
    mock_preflight.assert_not_called()


def test_dead_recorded_search_only_starts_search_only(tmp_path):
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="search-only")

    mock_do_start, _ = _run_start(reg, _start_args())

    assert mock_do_start.call_args.kwargs["search_only"] is True
    assert mock_do_start.call_args.kwargs["memory_only"] is False


def test_dead_recorded_full_still_starts_full(tmp_path):
    """A recorded `full` is inherited as `full` — the fix must not invent a
    memory-only default for workspaces that genuinely index."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="full")

    mock_do_start, mock_preflight = _run_start(reg, _start_args())

    assert mock_do_start.call_args.kwargs["memory_only"] is False
    assert mock_do_start.call_args.kwargs["search_only"] is False
    mock_preflight.assert_called_once()


def test_no_recorded_entry_starts_full(tmp_path):
    """First-ever start of a workspace: unchanged default, full mode."""
    reg = _registry(tmp_path)

    mock_do_start, _ = _run_start(reg, _start_args())

    assert mock_do_start.call_args.kwargs["memory_only"] is False
    assert mock_do_start.call_args.kwargs["search_only"] is False


def test_registry_entry_without_mode_field_starts_full(tmp_path):
    """Registry rows written by older vectr builds have no `mode` key; the
    missing field must fall back to full, not crash or guess."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999)  # register's default is full

    mock_do_start, _ = _run_start(reg, _start_args())

    assert mock_do_start.call_args.kwargs["memory_only"] is False
    assert mock_do_start.call_args.kwargs["search_only"] is False


# ---------------------------------------------------------------------------
# Explicit flags outrank the record
# ---------------------------------------------------------------------------

def test_explicit_memory_only_overrides_recorded_full(tmp_path):
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="full")

    mock_do_start, _ = _run_start(reg, _start_args(memory_only=True))

    assert mock_do_start.call_args.kwargs["memory_only"] is True


def test_explicit_search_only_overrides_recorded_memory_only(tmp_path):
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="memory-only")

    mock_do_start, _ = _run_start(reg, _start_args(search_only=True))

    assert mock_do_start.call_args.kwargs["search_only"] is True
    assert mock_do_start.call_args.kwargs["memory_only"] is False


def test_explicit_full_overrides_recorded_memory_only(tmp_path):
    """`--full` is the escape hatch: without it a workspace recorded
    memory-only could never be started as an indexer again."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="memory-only")

    mock_do_start, mock_preflight = _run_start(reg, _start_args(full=True))

    assert mock_do_start.call_args.kwargs["memory_only"] is False
    assert mock_do_start.call_args.kwargs["search_only"] is False
    mock_preflight.assert_called_once()


def test_full_flag_conflicts_with_mode_flags(tmp_path, capsys):
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="memory-only")

    with pytest.raises(SystemExit) as exc:
        _run_start(reg, _start_args(full=True, memory_only=True))

    assert exc.value.code == 1
    assert "--full cannot be combined" in capsys.readouterr().err


def test_mode_flags_still_mutually_exclusive(tmp_path, capsys):
    reg = _registry(tmp_path)

    with pytest.raises(SystemExit) as exc:
        _run_start(reg, _start_args(memory_only=True, search_only=True))

    assert exc.value.code == 1
    assert "mutually exclusive" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Inheritance is announced, and does not touch the record or other commands
# ---------------------------------------------------------------------------

def test_inherited_mode_is_announced_on_stderr(tmp_path, capsys):
    """A mode the user did not ask for must never be invisible."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="memory-only")

    _run_start(reg, _start_args())

    err = capsys.readouterr().err
    assert "memory-only" in err
    assert "inherited" in err
    assert "--full" in err


def test_explicit_mode_is_not_announced_as_inherited(tmp_path, capsys):
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="memory-only")

    _run_start(reg, _start_args(memory_only=True))

    assert "inherited" not in capsys.readouterr().err


def test_inherited_search_only_reaches_ide_config_writer(tmp_path):
    """The inherited mode must also drive the IDE config `start` writes — a
    search-only daemon advertises a different tool set (the same property the
    restart tests pin)."""
    ws = str(tmp_path)
    reg = _registry(tmp_path)
    reg.register(workspace_hash(ws), ws, 8765, 99999, mode="search-only")

    with patch("main.InstanceRegistry", return_value=reg), \
         patch("agent.instance_registry._is_pid_alive", return_value=False), \
         patch("main._is_pid_alive", return_value=False), \
         patch("agent.instance_registry._port_is_free", return_value=True), \
         patch("main._write_workspace_config") as mock_write, \
         patch("main._preflight_grammars"), \
         patch("main._do_start"):
        m.cmd_start(_start_args(paths=[ws]))

    assert mock_write.call_args.kwargs["search_only"] is True


def test_json_output_stdout_stays_json_only(tmp_path, capsys):
    """The announcement goes to stderr so `--json` stdout stays parseable."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 99999, mode="memory-only")

    _run_start(reg, _start_args(json_output=True))

    assert capsys.readouterr().out == ""


def test_dead_entry_is_still_pruned_after_start(tmp_path):
    """`prune_dead()` keeps deleting dead entries wholesale — `cmd_start` reads
    the mode BEFORE the sweep rather than weakening the sweep itself."""
    reg = _registry(tmp_path)
    wh = workspace_hash(WS)
    reg.register(wh, WS, 8765, 99999, mode="memory-only")

    _run_start(reg, _start_args())

    assert wh not in json.loads((tmp_path / "instances.json").read_text())


def test_live_recorded_entry_still_short_circuits(tmp_path, capsys):
    """A live instance is untouched: start reports 'already running' and never
    launches a second daemon, whatever its mode."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 12345, mode="memory-only")

    mock_do_start, _ = _run_start(reg, _start_args(), alive=True)

    mock_do_start.assert_not_called()
    assert "already running" in capsys.readouterr().err


def test_stopped_workspace_gets_no_ghost_inheritance(tmp_path):
    """`vectr stop` unregisters the workspace, so the next start is a genuine
    first start again — full mode, no inherited memory-only."""
    reg = _registry(tmp_path)
    wh = workspace_hash(WS)
    reg.register(wh, WS, 8765, 99999, mode="memory-only")
    reg.unregister(wh)

    mock_do_start, _ = _run_start(reg, _start_args())

    assert mock_do_start.call_args.kwargs["memory_only"] is False


def test_restart_mode_inheritance_unchanged(tmp_path):
    """Regression guard for pulling the rule into the shared
    `_resolve_inherited_mode`: `restart` still inherits the recorded mode."""
    reg = _registry(tmp_path)
    reg.register(workspace_hash(WS), WS, 8765, 12345, mode="memory-only")

    with patch("main.InstanceRegistry", return_value=reg), \
         patch("agent.instance_registry._port_is_free", return_value=True), \
         patch("agent.instance_registry._is_pid_alive", return_value=True), \
         patch("main._is_pid_alive", return_value=True), \
         patch("main._stop_server"), \
         patch("main._write_workspace_config"), \
         patch("main._do_start") as mock_do_start:
        m.cmd_restart(_start_args())

    assert mock_do_start.call_args.kwargs["memory_only"] is True
