"""Settings + localhost-only enforcement tests (UPG-PRO-6)."""
from __future__ import annotations

import pytest

from agent.proactive.settings import (
    ProactiveSettings,
    derive_provider_timeout_s,
    proactive_bind_is_loopback,
)


def test_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("VECTR_PROACTIVE"):
            monkeypatch.delenv(k, raising=False)
    s = ProactiveSettings.from_env()
    assert s.enabled is True  # UPG-PROXY-DEFAULT-ON-GATE
    assert s.proxy_inject is True
    assert s.cache_similarity_threshold == 1.0


def test_config_layer_defaults_proactive_enabled_true():
    # Pins the default independently of the env-override path above:
    # agent/config.yaml's proactive.enabled (surfaced as
    # agent.config.PROACTIVE_ENABLED) is the bundled default ProactiveSettings
    # .from_env() falls back to when VECTR_PROACTIVE is unset.
    from agent import config

    assert config.PROACTIVE_ENABLED is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VECTR_PROACTIVE", "1")
    monkeypatch.setenv("VECTR_PROACTIVE_MIN_SIMILARITY", "0.5")
    monkeypatch.setenv("VECTR_PROACTIVE_MAX_ITEMS", "7")
    monkeypatch.setenv("VECTR_PROACTIVE_PROXY_PORT", "19999")
    monkeypatch.setenv("VECTR_PROACTIVE_CACHE", "true")
    s = ProactiveSettings.from_env()
    assert s.enabled is True
    assert s.min_similarity == 0.5
    assert s.max_items_per_event == 7
    assert s.proxy_port == 19999
    assert s.cache_enabled is True


def test_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("VECTR_PROACTIVE_MAX_ITEMS", "not-an-int")
    s = ProactiveSettings.from_env()
    assert s.max_items_per_event == 3  # bundled default, not a crash


def test_cooldown_ttl_default_and_normalisation(monkeypatch):
    # UPG-PROXY-COOLDOWN-NO-TIME-DECAY: the shipped default enables one hour of
    # wall-clock decay on top of the count ring; a non-positive value means
    # "no time decay" and normalises to None (the SessionLedger's canonical
    # disabled sentinel — a literal 0 there would expire every entry
    # instantly, which is NOT the intended meaning of disabling).
    for k in list(__import__("os").environ):
        if k.startswith("VECTR_PROACTIVE"):
            monkeypatch.delenv(k, raising=False)
    s = ProactiveSettings.from_env()
    assert s.cooldown_ttl_seconds == 3600.0

    monkeypatch.setenv("VECTR_PROACTIVE_COOLDOWN_TTL_SECONDS", "120")
    assert ProactiveSettings.from_env().cooldown_ttl_seconds == 120.0

    for disabled in ("0", "-5"):
        monkeypatch.setenv("VECTR_PROACTIVE_COOLDOWN_TTL_SECONDS", disabled)
        assert ProactiveSettings.from_env().cooldown_ttl_seconds is None


def test_cooldown_ttl_bad_env_falls_back_to_default(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("VECTR_PROACTIVE"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VECTR_PROACTIVE_COOLDOWN_TTL_SECONDS", "not-a-number")
    assert ProactiveSettings.from_env().cooldown_ttl_seconds == 3600.0


# -- unconditional bind check (UPG-PROXY-LOOPBACK-BYPASS) -------------------


def test_bind_is_loopback_reads_the_plumbed_env_var(monkeypatch):
    monkeypatch.setenv("VECTR_BIND_HOST", "127.0.0.1")
    assert proactive_bind_is_loopback() is True
    monkeypatch.setenv("VECTR_BIND_HOST", "localhost")
    assert proactive_bind_is_loopback() is True
    monkeypatch.setenv("VECTR_BIND_HOST", "0.0.0.0")
    assert proactive_bind_is_loopback() is False
    monkeypatch.setenv("VECTR_BIND_HOST", "10.0.0.5")
    assert proactive_bind_is_loopback() is False


def test_bind_is_loopback_defaults_to_loopback_when_unset(monkeypatch):
    # Absent VECTR_BIND_HOST (in-process/test usage that never went through
    # main.py's `_do_start` subprocess spawn) must fall back to the
    # documented default bind, 127.0.0.1 — never be mistaken for
    # non-loopback and spuriously refuse in-process callers.
    monkeypatch.delenv("VECTR_BIND_HOST", raising=False)
    assert proactive_bind_is_loopback() is True


def test_bind_is_loopback_is_unconditional_no_config_argument():
    # This check takes no config argument at all — it cannot be skipped by a
    # config toggle, which is exactly the point (UPG-PROXY-LOOPBACK-BYPASS).
    import inspect

    sig = inspect.signature(proactive_bind_is_loopback)
    assert list(sig.parameters) == []


# -- argv --host fallback (a daemon started outside main.py's _do_start) ----


def test_bind_falls_back_to_argv_host_space_separated(monkeypatch):
    # A daemon launched directly (e.g. `uvicorn api:app --host 0.0.0.0`,
    # bypassing main.py's `_do_start`) has no VECTR_BIND_HOST but IS invoked
    # with the same --host argument uvicorn itself reads — this must not be
    # silently treated as loopback.
    monkeypatch.delenv("VECTR_BIND_HOST", raising=False)
    monkeypatch.setattr("sys.argv", ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"])
    assert proactive_bind_is_loopback() is False


def test_bind_falls_back_to_argv_host_equals_form(monkeypatch):
    monkeypatch.delenv("VECTR_BIND_HOST", raising=False)
    monkeypatch.setattr("sys.argv", ["uvicorn", "api:app", "--host=0.0.0.0"])
    assert proactive_bind_is_loopback() is False


def test_bind_argv_host_loopback_value_still_loopback(monkeypatch):
    monkeypatch.delenv("VECTR_BIND_HOST", raising=False)
    monkeypatch.setattr("sys.argv", ["uvicorn", "api:app", "--host", "127.0.0.1"])
    assert proactive_bind_is_loopback() is True


def test_bind_env_var_takes_precedence_over_argv(monkeypatch):
    # Resolution order: VECTR_BIND_HOST, then argv --host, then 127.0.0.1.
    monkeypatch.setenv("VECTR_BIND_HOST", "127.0.0.1")
    monkeypatch.setattr("sys.argv", ["uvicorn", "api:app", "--host", "0.0.0.0"])
    assert proactive_bind_is_loopback() is True


def test_bind_absent_env_and_absent_argv_defaults_to_loopback(monkeypatch):
    # Neither VECTR_BIND_HOST nor a --host in argv (ordinary in-process/test
    # usage, e.g. this very pytest invocation) — must resolve to loopback,
    # not be mistaken for non-loopback.
    monkeypatch.delenv("VECTR_BIND_HOST", raising=False)
    monkeypatch.setattr("sys.argv", ["pytest", "tests/test_proactive_settings.py"])
    assert proactive_bind_is_loopback() is True


# -- provider-timeout / outer-budget ordering invariant (UPG-PROXY-BUDGET-40MS) --


def _settings(**over):
    """Full ProactiveSettings from the bundled defaults, with fields overridden
    for the ordering test — avoids repeating every field just to vary one or two."""
    import dataclasses

    return dataclasses.replace(ProactiveSettings.from_env(), **over)


def test_bundled_default_derives_below_budget(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("VECTR_PROACTIVE"):
            monkeypatch.delenv(k, raising=False)
    # The shipped default (config.yaml) must already satisfy the invariant.
    s = ProactiveSettings.from_env()
    derived = derive_provider_timeout_s(s)
    budget_s = s.proxy_inject_budget_ms / 1000.0
    assert derived < budget_s
    assert s.proxy_inject_budget_ms >= 750  # UPG-PROXY-BUDGET-40MS: no longer self-defeating


def test_derived_timeout_strictly_below_budget_across_configs():
    for budget_ms, fraction, cap_s in [
        (750, 0.8, 2.0),
        (40, 0.8, 2.0),   # a tight, misconfigured-small budget
        (100, 1.0, 100.0),  # fraction/cap misconfigured to be permissive
        (5000, 0.5, 0.2),  # cap binds well below fraction * budget
    ]:
        s = _settings(
            proxy_inject_budget_ms=budget_ms,
            proxy_inject_provider_timeout_fraction=fraction,
            proxy_inject_provider_timeout_max_s=cap_s,
        )
        derived = derive_provider_timeout_s(s)
        budget_s = max(budget_ms, 1) / 1000.0
        assert derived < budget_s, (budget_ms, fraction, cap_s, derived)
        assert derived > 0


def test_derived_timeout_respects_fraction_and_cap():
    s = _settings(
        proxy_inject_budget_ms=1000, proxy_inject_provider_timeout_fraction=0.5,
        proxy_inject_provider_timeout_max_s=2.0,
    )
    # fraction * budget (0.5s) is below the cap (2.0s), so fraction governs.
    assert derive_provider_timeout_s(s) == pytest.approx(0.5)

    s = _settings(
        proxy_inject_budget_ms=10_000, proxy_inject_provider_timeout_fraction=0.9,
        proxy_inject_provider_timeout_max_s=1.0,
    )
    # fraction * budget (9s) exceeds the cap (1.0s), so the cap governs.
    assert derive_provider_timeout_s(s) == pytest.approx(1.0)
