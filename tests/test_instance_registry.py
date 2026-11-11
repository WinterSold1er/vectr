"""Tests for agent/instance_registry.py."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.instance_registry import (
    InstanceRegistry,
    PortBusyError,
    _is_pid_alive,
    _port_is_free,
    workspace_hash,
)


# ---------------------------------------------------------------------------
# workspace_hash
# ---------------------------------------------------------------------------

def test_workspace_hash_is_12_hex_chars():
    h = workspace_hash("/some/project")
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_workspace_hash_is_deterministic():
    assert workspace_hash("/foo/bar") == workspace_hash("/foo/bar")


def test_workspace_hash_differs_for_different_paths():
    assert workspace_hash("/project/a") != workspace_hash("/project/b")


# ---------------------------------------------------------------------------
# InstanceRegistry — basic CRUD
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    return InstanceRegistry(registry_path=tmp_path / "instances.json")


def test_empty_registry_get_returns_none(registry):
    assert registry.get("aabbccddeeff") is None


def test_empty_registry_list_all_returns_empty(registry):
    assert registry.list_all() == {}


def test_register_and_get(registry):
    registry.register("abc123456789", "/project/a", 8765, 12345)
    entry = registry.get("abc123456789")
    assert entry is not None
    assert entry["workspace"] == "/project/a"
    assert entry["port"] == 8765
    assert entry["pid"] == 12345
    assert "started_at" in entry


def test_register_multiple_workspaces(registry):
    registry.register("aaa000000000", "/project/a", 8765, 100)
    registry.register("bbb000000000", "/project/b", 8766, 101)
    all_entries = registry.list_all()
    assert len(all_entries) == 2
    assert all_entries["aaa000000000"]["port"] == 8765
    assert all_entries["bbb000000000"]["port"] == 8766


def test_unregister_removes_entry(registry):
    registry.register("abc123456789", "/project/a", 8765, 12345)
    registry.unregister("abc123456789")
    assert registry.get("abc123456789") is None


def test_unregister_nonexistent_is_noop(registry):
    registry.unregister("doesnotexist")  # should not raise


def test_register_defaults_extra_roots_and_code_workspace_file(registry):
    registry.register("abc123456789", "/project/a", 8765, 12345)
    entry = registry.get("abc123456789")
    assert entry["extra_roots"] == []
    assert entry["code_workspace_file"] is None


def test_register_stores_extra_roots_and_code_workspace_file(registry):
    registry.register(
        "abc123456789", "/project/a", 8765, 12345,
        extra_roots=["/project/b"], code_workspace_file="/project/proj.code-workspace",
    )
    entry = registry.get("abc123456789")
    assert entry["extra_roots"] == ["/project/b"]
    assert entry["code_workspace_file"] == "/project/proj.code-workspace"


def test_register_defaults_mode_and_host(registry):
    """UPG-RESTART-DROPS-MODE: a plain register records the plain launch —
    full mode on the loopback bind — so `restart` inherits something truthful
    even from an entry written by a caller that names neither."""
    registry.register("abc123456789", "/project/a", 8765, 12345)
    entry = registry.get("abc123456789")
    assert entry["mode"] == "full"
    assert entry["host"] == "127.0.0.1"


def test_register_stores_mode_and_host(registry):
    registry.register(
        "abc123456789", "/project/a", 8765, 12345, mode="memory-only", host="0.0.0.0",
    )
    entry = registry.get("abc123456789")
    assert entry["mode"] == "memory-only"
    assert entry["host"] == "0.0.0.0"


def test_register_overwrites_existing_entry(registry):
    registry.register("abc123456789", "/project/a", 8765, 100)
    registry.register("abc123456789", "/project/a", 8766, 200)
    entry = registry.get("abc123456789")
    assert entry["port"] == 8766
    assert entry["pid"] == 200


# ---------------------------------------------------------------------------
# InstanceRegistry — prune_dead
# ---------------------------------------------------------------------------

def test_prune_dead_removes_dead_pid(registry):
    with patch("agent.instance_registry._is_pid_alive", return_value=False):
        registry.register("aaa000000000", "/project/a", 8765, 99999)
        registry.prune_dead()
    assert registry.get("aaa000000000") is None


def test_prune_dead_keeps_live_pid(registry):
    with patch("agent.instance_registry._is_pid_alive", return_value=True):
        registry.register("aaa000000000", "/project/a", 8765, 99999)
        registry.prune_dead()
    assert registry.get("aaa000000000") is not None


def test_prune_dead_only_removes_dead_entries(registry):
    registry.register("aaa000000000", "/project/a", 8765, 11111)
    registry.register("bbb000000000", "/project/b", 8766, 22222)

    def side_effect(pid):
        return pid == 11111  # only first is alive

    with patch("agent.instance_registry._is_pid_alive", side_effect=side_effect):
        registry.prune_dead()

    assert registry.get("aaa000000000") is not None
    assert registry.get("bbb000000000") is None


# ---------------------------------------------------------------------------
# InstanceRegistry — atomic write
# ---------------------------------------------------------------------------

def test_write_is_atomic(registry):
    """Written file must be valid JSON (no partial writes)."""
    registry.register("abc123456789", "/project/a", 8765, 12345)
    content = registry._path.read_text()
    parsed = json.loads(content)
    assert "abc123456789" in parsed


def test_no_tmp_file_left_after_write(registry):
    registry.register("abc123456789", "/project/a", 8765, 12345)
    assert not registry._path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# InstanceRegistry — find_free_port
# ---------------------------------------------------------------------------

def test_find_free_port_returns_live_entry_port(registry):
    """If the entry is alive, return its port (caller must detect no-op)."""
    registry.register("aaa000000000", "/project/a", 8765, 11111)
    with patch("agent.instance_registry._is_pid_alive", return_value=True):
        port = registry.find_free_port("aaa000000000", 8765)
    assert port == 8765


def test_find_free_port_reuses_dead_entry_port_if_free(registry):
    """Dead entry → reuse its previous port (avoids rewriting .mcp.json)."""
    registry.register("aaa000000000", "/project/a", 8900, 99999)

    def pid_dead(pid):
        return False

    def port_free(port):
        return True  # old port is free

    with patch("agent.instance_registry._is_pid_alive", side_effect=pid_dead), \
         patch("agent.instance_registry._port_is_free", side_effect=port_free):
        port = registry.find_free_port("aaa000000000", 8765)

    assert port == 8900  # reuses the previously assigned port


def test_find_free_port_scans_forward_when_dead_port_taken(registry):
    """Dead entry with its old port taken → scan forward from preferred."""
    registry.register("aaa000000000", "/project/a", 8765, 99999)

    def pid_dead(pid):
        return False

    # Old port (8765) is taken, 8766 is free
    def port_free(port):
        return port == 8766

    with patch("agent.instance_registry._is_pid_alive", side_effect=pid_dead), \
         patch("agent.instance_registry._port_is_free", side_effect=port_free):
        port = registry.find_free_port("aaa000000000", 8765)

    assert port == 8766


def test_find_free_port_new_workspace_finds_first_free(registry):
    """Unknown workspace → scan from preferred_port."""
    def port_free(port):
        return port == 8767  # only 8767 is free

    with patch("agent.instance_registry._port_is_free", side_effect=port_free):
        port = registry.find_free_port("newworkspace0", 8765)

    assert port == 8767


def test_find_free_port_raises_when_none_available(registry):
    with patch("agent.instance_registry._port_is_free", return_value=False):
        with pytest.raises(RuntimeError, match="No free port"):
            registry.find_free_port("newworkspace0", 8765)


def test_find_free_port_retries_previous_port_before_walking_forward(registry):
    """UPG-RESTART-PORT-WALK-BREAKS-MCP: the previous port can still be
    mid-teardown for a couple of probes right after a `vectr stop` even with
    the SO_REUSEADDR-aware probe (agent/instance_registry.py's
    `_port_is_free`) — reuse must retry a bounded number of times before
    giving up and walking to a different port, not bail on the first miss."""
    registry.register("aaa000000000", "/project/a", 8900, 99999)

    calls: list[int] = []

    def port_free(port: int) -> bool:
        calls.append(port)
        # 8900 looks busy for the first two probes, then clears.
        if port == 8900:
            return calls.count(8900) >= 3
        return False

    with patch("agent.instance_registry._is_pid_alive", return_value=False), \
         patch("agent.instance_registry._port_is_free", side_effect=port_free), \
         patch("agent.instance_registry.time.sleep") as mock_sleep:
        port = registry.find_free_port("aaa000000000", 8765)

    assert port == 8900  # reused, never walked to 8766+
    assert calls.count(8900) == 3
    assert mock_sleep.call_count == 2  # slept between the 3 attempts, not after the last


def test_find_free_port_strict_returns_preferred_when_free(registry):
    """strict=True, unknown workspace, preferred port free → return it
    immediately without scanning forward (UPG-CLI-STOP-NO-PORT-FLAG)."""
    calls: list[int] = []

    def port_free(port: int) -> bool:
        calls.append(port)
        return port == 8765

    with patch("agent.instance_registry._port_is_free", side_effect=port_free):
        port = registry.find_free_port("newworkspace0", 8765, strict=True)

    assert port == 8765
    assert calls == [8765]  # never probed 8766+


def test_find_free_port_strict_raises_when_preferred_busy(registry):
    """strict=True, unknown workspace, preferred port busy → PortBusyError,
    never silently walk to a different port."""
    with patch("agent.instance_registry._port_is_free", return_value=False):
        with pytest.raises(PortBusyError, match="8765"):
            registry.find_free_port("newworkspace0", 8765, strict=True)


def test_find_free_port_strict_raises_after_dead_entry_retries_exhausted(registry):
    """strict=True, dead entry whose previous port never frees up within the
    bounded retry window → PortBusyError naming that previous port, not a
    silent walk to preferred_port or beyond."""
    registry.register("aaa000000000", "/project/a", 8900, 99999)

    with patch("agent.instance_registry._is_pid_alive", return_value=False), \
         patch("agent.instance_registry._port_is_free", return_value=False), \
         patch("agent.instance_registry.time.sleep"):
        with pytest.raises(PortBusyError, match="8900"):
            registry.find_free_port("aaa000000000", 8765, strict=True)


def test_find_free_port_strict_reuses_live_dead_entry_port_without_error(registry):
    """strict=True, dead entry whose previous port IS free → reuse it, same
    as non-strict mode; strict only changes the no-port-available path."""
    registry.register("aaa000000000", "/project/a", 8900, 99999)

    with patch("agent.instance_registry._is_pid_alive", return_value=False), \
         patch("agent.instance_registry._port_is_free", return_value=True):
        port = registry.find_free_port("aaa000000000", 8765, strict=True)

    assert port == 8900


def test_find_free_port_retry_is_bounded_by_config(registry):
    """After INSTANCE_REGISTRY_PORT_REUSE_RETRY_ATTEMPTS failed reuse probes,
    give up on the previous port and fall through to the forward scan —
    reuse must not retry forever."""
    # Sourced from agent.config directly (not agent.instance_registry): the
    # module imports this lazily inside find_free_port() to keep it off the
    # agent.hook_cli fast-dispatch import path (see the comment there).
    from agent.config import INSTANCE_REGISTRY_PORT_REUSE_RETRY_ATTEMPTS

    registry.register("aaa000000000", "/project/a", 8900, 99999)

    calls: list[int] = []

    def port_free(port: int) -> bool:
        calls.append(port)
        return port == 8766  # 8900 never frees up; 8766 (forward scan) does

    with patch("agent.instance_registry._is_pid_alive", return_value=False), \
         patch("agent.instance_registry._port_is_free", side_effect=port_free), \
         patch("agent.instance_registry.time.sleep"):
        port = registry.find_free_port("aaa000000000", 8765)

    assert port == 8766
    assert calls.count(8900) == INSTANCE_REGISTRY_PORT_REUSE_RETRY_ATTEMPTS


# ---------------------------------------------------------------------------
# _is_pid_alive / _port_is_free (module-level helpers)
# ---------------------------------------------------------------------------

def test_is_pid_alive_current_process():
    assert _is_pid_alive(os.getpid()) is True


def test_is_pid_alive_nonexistent_pid():
    # PID 0 is not a real user process; os.kill(0, 0) sends to the process group.
    # Use a very high PID that is almost certainly dead.
    assert _is_pid_alive(99999999) is False


def test_port_is_free_high_port():
    # A fixed high port (e.g. 59876) races the OS's own ephemeral port
    # allocator — another process on the machine can be bound to it when
    # this test runs (UPG-TEST-PORT-FLAKE), so assert on a port the OS itself
    # just certified as free instead of a hardcoded guess: bind a throwaway
    # socket to port 0 (kernel picks any free port), read back the assigned
    # port, then release it and confirm _port_is_free agrees.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert _port_is_free(free_port) is True


def test_port_is_free_false_when_port_is_held():
    # Companion case: a port this test itself holds open must report False.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        held_port = held.getsockname()[1]
        assert _port_is_free(held_port) is False


def _put_port_in_time_wait(port_hint: int = 0) -> int:
    """Drive a real 127.0.0.1 TCP port into the OS's post-close linger state
    the way an HTTP daemon actually does: accept a connection, exchange data,
    then close from the server side first. Returns the port. Uses `port_hint`
    (default 0 = kernel-assigned ephemeral) so callers never hardcode a port
    that could collide with a live daemon."""
    from tests.conftest import allow_loopback_port

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port_hint))
    srv.listen(1)
    port = srv.getsockname()[1]

    def _client() -> None:
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect(("127.0.0.1", port))
        c.send(b"hi")
        time.sleep(0.2)
        c.close()

    # The client dials this test's OWN just-bound listener — register it with
    # the suite-wide socket guard (tests/conftest.py) for exactly the dance's
    # duration, like every other test-owned loopback server.
    with allow_loopback_port(port):
        t = threading.Thread(target=_client)
        t.start()
        conn, _addr = srv.accept()
        conn.recv(2)
        conn.close()  # server side closes first -> this port lands in TIME_WAIT
        t.join()
    srv.close()
    time.sleep(0.05)
    return port


def test_port_is_free_true_for_time_wait_port():
    """UPG-RESTART-PORT-WALK-BREAKS-MCP: a port whose listener just closed an
    established connection sits in TIME_WAIT for the OS's linger window —
    without SO_REUSEADDR on the probe socket, `_port_is_free` used to report
    this as busy even though the only kind of bind vectr's own daemon ever
    performs (uvicorn, which always sets SO_REUSEADDR) would succeed on it
    immediately. This is a real socket dance, not a mock: it reproduces the
    exact kernel state a `vectr stop` + `start` a moment later hits."""
    port = _put_port_in_time_wait()
    assert _port_is_free(port) is True


def test_find_free_port_reuses_time_wait_port_end_to_end(registry):
    """End-to-end (real registry + real TIME_WAIT socket, no mocking of
    `_port_is_free`): a dead entry's previous port, currently in TIME_WAIT,
    must be reused rather than walked past."""
    port = _put_port_in_time_wait()
    registry.register("aaa000000000", "/project/a", port, 99999999)  # dead pid

    got = registry.find_free_port("aaa000000000", port)

    assert got == port


# ---------------------------------------------------------------------------
# UPG-TEST-REGISTRY-NOT-ISOLATED: a unit test must never see the developer's
# real instance registry. The session-scoped `_isolated_instance_registry`
# fixture (tests/conftest.py) wraps `InstanceRegistry.__init__` so a no-arg
# `InstanceRegistry()` lands in a per-session tmp dir instead of
# `Path.home() / ".vectr" / "instances.json`. This block pins that property:
# (a) the no-arg default is NOT the developer's real path, and (b) tests
# that pass an explicit `registry_path=` still get the path they asked for
# (the wrapper does not silently redirect explicit callers).
# ---------------------------------------------------------------------------

def test_no_arg_default_is_not_developers_real_registry(tmp_path):
    """`InstanceRegistry()` with no args must point at a per-session tmp
    file, never at the developer's real `~/.vectr/instances.json`. A test
    that builds the no-arg form and reads/writes it must not see the live
    daemon's entry on port 8765 and must not corrupt the user's real file."""
    from agent import instance_registry

    real_default = instance_registry.REGISTRY_PATH

    reg = InstanceRegistry()
    assert reg._path != real_default, (
        f"no-arg InstanceRegistry() fell through to the developer's real "
        f"registry ({reg._path}); the _isolated_instance_registry fixture "
        f"in tests/conftest.py did not redirect it."
    )
    # And: the redirected path must NOT live under Path.home()/.vectr/ — the
    # whole point of the fixture is to keep the unit run off the developer's
    # home tree.
    home_vectr = Path.home() / ".vectr"
    assert not (reg._path == home_vectr / "instances.json"), (
        f"no-arg InstanceRegistry() still resolves under {home_vectr}; the "
        f"session isolation did not engage."
    )


def test_explicit_registry_path_still_honored(tmp_path):
    """An explicit `registry_path=` from a test or product caller must
    win — the wrapper only substitutes when the caller passed nothing. A
    test that names its own tmp_path must keep that path, not be silently
    redirected to the session-default location."""
    explicit = tmp_path / "explicit.json"
    reg = InstanceRegistry(registry_path=explicit)
    assert reg._path == explicit, (
        f"explicit registry_path= was overridden: expected {explicit}, "
        f"got {reg._path}"
    )


def test_no_arg_register_and_read_stays_within_isolated_path(tmp_path):
    """End-to-end pin: a no-arg `InstanceRegistry()` registers an entry,
    re-reads it, and the on-disk file is the redirected (tmp) path — not
    the developer's real one. Proves the fixture covers the read+write
    surface, not just the constructor."""
    reg = InstanceRegistry()
    reg.register("abc123456789", "/project/a", 8765, 12345)
    # The write went to the redirected path, not the real one.
    assert reg._path.exists(), f"expected write at {reg._path}"
    real_path = Path.home() / ".vectr" / "instances.json"
    assert reg._path != real_path, (
        f"a unit test wrote to the developer's real registry ({real_path}); "
        f"isolation failed."
    )
