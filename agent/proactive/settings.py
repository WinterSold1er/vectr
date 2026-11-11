"""Runtime settings for proactive context (UPG-PRO-6 / UPG-PRO-16).

Layers deployment/runtime env vars over the bundled `config.yaml` defaults
(exposed as constants in `agent/config.py`), matching vectr's established split:
product-behaviour defaults live in yaml, runtime toggles in env. Nothing here is
persisted.

Also owns the localhost-only enforcement (design §10). It is UNCONDITIONAL:
`proactive_bind_is_loopback` is the runtime gate inside `proactive_context`,
consulted before any config/channel check, so a non-loopback bind refuses
proactive injection for every channel — the proxy channel included — regardless
of the master switch or a client-supplied `channel` label. Proactive context
reads the conversation, the most sensitive data on the machine, so this refusal
cannot be a config toggle a caller can route around.

UPG-PROACTIVE-DEAD-GATES removed two CONFIG-GATED siblings that had zero
production call sites (`enforce_proactive_bind`, `proactive_enabled`, their
`ProactiveRefused`, and the `proxy.enabled` config surface behind them): with
the master switch defaulting on they could not distinguish opt-in from
default-on, and the unconditional check above already enforces the same
boundary at every runtime seam. A non-loopback bind with proactive left off is
still a legitimate startup — nothing refuses it; only whether proactive
context is ever served changes.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from agent import config as _c


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _ttl_or_none(seconds: float) -> float | None:
    """Normalise the cooldown-TTL knob (UPG-PROXY-COOLDOWN-NO-TIME-DECAY).

    Non-positive means "no time decay" — the historical pure-count cooldown
    ring — and is normalised to `None` so the gate keeps one canonical
    disabled representation. (`SessionLedger` treats only `None` as
    decay-disabled; a literal 0 there would expire every entry instantly,
    which is a different — and never-intended — behaviour.)
    """
    return seconds if seconds > 0 else None


@dataclass(frozen=True)
class ProactiveSettings:
    """Fully-resolved proactive-context settings (env over yaml defaults)."""

    enabled: bool
    min_similarity: float
    max_items_per_event: int
    max_chars_per_event: int
    cooldown_items: int
    # UPG-PROXY-COOLDOWN-NO-TIME-DECAY: wall-clock suppression window for the
    # SessionLedger cooldown ring, in seconds. An anchor stops suppressing once
    # its last delivery is older than this, instead of staying suppressed until
    # `cooldown_items` OTHER distinct anchors have cycled through the ring.
    # None (normalised from any non-positive value) disables the decay entirely
    # — the historical count-only behaviour. Wired into the gate via
    # ProactiveGate(cooldown_ttl_seconds=...) in app/service.py.
    cooldown_ttl_seconds: float | None

    matcher_structural_note: bool
    matcher_semantic_note: bool
    matcher_code_search: bool

    proxy_host: str
    proxy_port: int
    proxy_upstream_base_url: str
    proxy_connect_timeout_s: float
    proxy_read_timeout_s: float
    proxy_inject: bool
    proxy_inject_budget_ms: int
    proxy_inject_provider_timeout_fraction: float
    proxy_inject_provider_timeout_max_s: float
    proxy_exclude_directive_notes: bool

    cache_enabled: bool
    cache_max_entries: int
    cache_ttl_seconds: float
    cache_similarity_threshold: float

    response_cache_enabled: bool
    response_cache_ttl_seconds: float
    response_cache_max_entries: int

    # UPG-PROXY-INJECT-PRECISION: structural-channel eligibility/overfetch/
    # scoring knobs (levers 1, 1b, 2, 3 — see agent/config.yaml's
    # `proactive.structural_*` / `proactive.max_weak_structural_items` block).
    structural_kinds: tuple[str, ...]
    structural_overfetch_multiplier: int
    structural_overfetch_ceiling: int
    structural_score_declared_anchor: float
    structural_score_declared_trigger: float
    structural_score_gotcha_mention: float
    structural_score_mention: float
    max_weak_structural_items: int

    @classmethod
    def from_env(cls) -> "ProactiveSettings":
        return cls(
            enabled=_env_bool("VECTR_PROACTIVE", _c.PROACTIVE_ENABLED),
            min_similarity=_env_float(
                "VECTR_PROACTIVE_MIN_SIMILARITY", _c.PROACTIVE_MIN_SIMILARITY
            ),
            max_items_per_event=_env_int(
                "VECTR_PROACTIVE_MAX_ITEMS", _c.PROACTIVE_MAX_ITEMS_PER_EVENT
            ),
            max_chars_per_event=_env_int(
                "VECTR_PROACTIVE_MAX_CHARS", _c.PROACTIVE_MAX_CHARS_PER_EVENT
            ),
            cooldown_items=_env_int(
                "VECTR_PROACTIVE_COOLDOWN", _c.PROACTIVE_COOLDOWN_ITEMS
            ),
            cooldown_ttl_seconds=_ttl_or_none(
                _env_float(
                    "VECTR_PROACTIVE_COOLDOWN_TTL_SECONDS",
                    _c.PROACTIVE_COOLDOWN_TTL_SECONDS,
                )
            ),
            matcher_structural_note=_env_bool(
                "VECTR_PROACTIVE_MATCH_STRUCTURAL", _c.PROACTIVE_MATCHER_STRUCTURAL_NOTE
            ),
            matcher_semantic_note=_env_bool(
                "VECTR_PROACTIVE_MATCH_SEMANTIC", _c.PROACTIVE_MATCHER_SEMANTIC_NOTE
            ),
            matcher_code_search=_env_bool(
                "VECTR_PROACTIVE_MATCH_CODE", _c.PROACTIVE_MATCHER_CODE_SEARCH
            ),
            proxy_host=_env_str("VECTR_PROACTIVE_PROXY_HOST", _c.PROACTIVE_PROXY_HOST),
            proxy_port=_env_int("VECTR_PROACTIVE_PROXY_PORT", _c.PROACTIVE_PROXY_PORT),
            proxy_upstream_base_url=_env_str(
                "VECTR_PROACTIVE_PROXY_UPSTREAM", _c.PROACTIVE_PROXY_UPSTREAM_BASE_URL
            ),
            proxy_connect_timeout_s=_env_float(
                "VECTR_PROACTIVE_PROXY_CONNECT_TIMEOUT", _c.PROACTIVE_PROXY_CONNECT_TIMEOUT_S
            ),
            proxy_read_timeout_s=_env_float(
                "VECTR_PROACTIVE_PROXY_READ_TIMEOUT", _c.PROACTIVE_PROXY_READ_TIMEOUT_S
            ),
            proxy_inject=_env_bool("VECTR_PROACTIVE_PROXY_INJECT", _c.PROACTIVE_PROXY_INJECT),
            proxy_inject_budget_ms=_env_int(
                "VECTR_PROACTIVE_PROXY_INJECT_BUDGET_MS", _c.PROACTIVE_PROXY_INJECT_BUDGET_MS
            ),
            proxy_inject_provider_timeout_fraction=_env_float(
                "VECTR_PROACTIVE_PROXY_INJECT_PROVIDER_TIMEOUT_FRACTION",
                _c.PROACTIVE_PROXY_INJECT_PROVIDER_TIMEOUT_FRACTION,
            ),
            proxy_inject_provider_timeout_max_s=_env_float(
                "VECTR_PROACTIVE_PROXY_INJECT_PROVIDER_TIMEOUT_MAX_S",
                _c.PROACTIVE_PROXY_INJECT_PROVIDER_TIMEOUT_MAX_S,
            ),
            proxy_exclude_directive_notes=_env_bool(
                "VECTR_PROACTIVE_PROXY_EXCLUDE_DIRECTIVE_NOTES",
                _c.PROACTIVE_PROXY_EXCLUDE_DIRECTIVE_NOTES,
            ),
            cache_enabled=_env_bool("VECTR_PROACTIVE_CACHE", _c.PROACTIVE_CACHE_ENABLED),
            cache_max_entries=_env_int(
                "VECTR_PROACTIVE_CACHE_MAX_ENTRIES", _c.PROACTIVE_CACHE_MAX_ENTRIES
            ),
            cache_ttl_seconds=_env_float(
                "VECTR_PROACTIVE_CACHE_TTL", _c.PROACTIVE_CACHE_TTL_SECONDS
            ),
            cache_similarity_threshold=_env_float(
                "VECTR_PROACTIVE_CACHE_SIMILARITY", _c.PROACTIVE_CACHE_SIMILARITY_THRESHOLD
            ),
            response_cache_enabled=_env_bool(
                "VECTR_PROACTIVE_RESPONSE_CACHE", _c.PROACTIVE_RESPONSE_CACHE_ENABLED
            ),
            response_cache_ttl_seconds=_env_float(
                "VECTR_PROACTIVE_RESPONSE_CACHE_TTL", _c.PROACTIVE_RESPONSE_CACHE_TTL_SECONDS
            ),
            response_cache_max_entries=_env_int(
                "VECTR_PROACTIVE_RESPONSE_CACHE_MAX_ENTRIES",
                _c.PROACTIVE_RESPONSE_CACHE_MAX_ENTRIES,
            ),
            structural_kinds=_env_csv(
                "VECTR_PROACTIVE_STRUCTURAL_KINDS", _c.PROACTIVE_STRUCTURAL_KINDS
            ),
            structural_overfetch_multiplier=_env_int(
                "VECTR_PROACTIVE_STRUCTURAL_OVERFETCH_MULTIPLIER",
                _c.PROACTIVE_STRUCTURAL_OVERFETCH_MULTIPLIER,
            ),
            structural_overfetch_ceiling=_env_int(
                "VECTR_PROACTIVE_STRUCTURAL_OVERFETCH_CEILING",
                _c.PROACTIVE_STRUCTURAL_OVERFETCH_CEILING,
            ),
            structural_score_declared_anchor=_env_float(
                "VECTR_PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR",
                _c.PROACTIVE_STRUCTURAL_SCORE_DECLARED_ANCHOR,
            ),
            structural_score_declared_trigger=_env_float(
                "VECTR_PROACTIVE_STRUCTURAL_SCORE_DECLARED_TRIGGER",
                _c.PROACTIVE_STRUCTURAL_SCORE_DECLARED_TRIGGER,
            ),
            structural_score_gotcha_mention=_env_float(
                "VECTR_PROACTIVE_STRUCTURAL_SCORE_GOTCHA_MENTION",
                _c.PROACTIVE_STRUCTURAL_SCORE_GOTCHA_MENTION,
            ),
            structural_score_mention=_env_float(
                "VECTR_PROACTIVE_STRUCTURAL_SCORE_MENTION",
                _c.PROACTIVE_STRUCTURAL_SCORE_MENTION,
            ),
            max_weak_structural_items=_env_int(
                "VECTR_PROACTIVE_MAX_WEAK_STRUCTURAL_ITEMS",
                _c.PROACTIVE_MAX_WEAK_STRUCTURAL_ITEMS,
            ),
        )


def derive_provider_timeout_s(settings: ProactiveSettings) -> float:
    """The daemon-provider's own httpx timeout for one injection round trip.

    Two timeouts guard `_maybe_inject`: this one, inside the provider's httpx
    call, and the proxy's outer `asyncio.wait_for(budget)` backstop. They must
    not be equal or inverted — if the outer backstop can fire at or before the
    provider's own timeout, `asyncio.wait_for` cancels the coroutine mid-flight
    on every slow daemon response, which the proxy can only record as an abrupt
    bypass (`inject_bypassed_error`, logged as a WARNING). If the provider's own
    timeout fires first instead, its `except` clause returns a clean
    `InjectionResult.empty()` — the proxy records a graceful `inject_skipped`
    with no warning, and forwarding still fails open.

    So the provider timeout is always derived as a strict fraction of the
    outer budget (config: `inject_provider_timeout_fraction`), capped at a
    sensible absolute ceiling (`inject_provider_timeout_max_s`) so a large
    budget never produces an unreasonably long per-request stall, and clamped
    a final time so it is always strictly below the outer budget regardless of
    how fraction/cap are configured.
    """
    budget_s = max(settings.proxy_inject_budget_ms, 1) / 1000.0
    derived = min(
        budget_s * settings.proxy_inject_provider_timeout_fraction,
        settings.proxy_inject_provider_timeout_max_s,
    )
    # Invariant clamp: strictly below the outer backstop no matter what
    # fraction/cap are configured to.
    return min(derived, budget_s * 0.95)


def _is_loopback(host: str) -> bool:
    """Loopback check — reuses the daemon's bind-guard helper rather than
    forking a second implementation (imported lazily to avoid an import cycle
    with main.py, which imports this package for `vectr proxy`)."""
    from main import _is_loopback_host

    return _is_loopback_host(host)


def _bind_host_from_argv() -> str | None:
    """Deterministically parse a `--host <value>` / `--host=<value>` pair out
    of the CURRENT PROCESS's own `sys.argv`, or None if absent.

    Plain positional argument parsing, not a query-content heuristic — this
    reads this process's own launch invocation, never a query or a config
    value. Handles both the space-separated and `=`-joined uvicorn/argparse
    forms. In-process/test usage (pytest, the MCP server) has no `--host` in
    its own argv, so this always resolves to None there — no test churn."""
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--host" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--host="):
            return arg[len("--host="):]
    return None


def _current_bind_host() -> str:
    """The daemon's actual bind host for THIS process.

    Resolution order:
    1. `VECTR_BIND_HOST` — `main.py`'s `_do_start` sets this on the uvicorn
       subprocess env to the exact `--host` it launched with
       (UPG-PROXY-LOOPBACK-BYPASS).
    2. This process's own `sys.argv` `--host` — covers a daemon started
       outside `_do_start` (e.g. `uvicorn api:app --host 0.0.0.0` run
       directly), which has no VECTR_BIND_HOST but IS launched with the same
       `--host` argument uvicorn itself reads; without this fallback such a
       daemon would be silently treated as loopback and would still serve
       proactive injection over a non-loopback bind.
    3. `127.0.0.1` — `_do_start`'s own documented default bind. Only reached
       when neither of the above is present (in-process/test usage, or the
       MCP server sharing the daemon's process) — never treat a fully
       unspecified bind as non-loopback, or every in-process caller would be
       spuriously refused."""
    env_host = os.environ.get("VECTR_BIND_HOST", "").strip()
    if env_host:
        return env_host
    argv_host = _bind_host_from_argv()
    if argv_host:
        return argv_host
    return "127.0.0.1"


def proactive_bind_is_loopback() -> bool:
    """Unconditional, channel-independent bind check for the proactive
    injection gate (UPG-PROXY-LOOPBACK-BYPASS).

    This takes no config argument and must never be skipped by one — it is
    the runtime refusal `proactive_context` consults BEFORE any config or
    `channel` branch, so a non-loopback bind refuses proactive injection —
    including the proxy channel's own launch-is-consent exemption from the
    master switch — for every caller. A non-loopback bind with proactive
    left off entirely is still a legitimate startup; this function only
    governs whether proactive context is ever served, not whether the daemon
    may start."""
    return _is_loopback(_current_bind_host())
