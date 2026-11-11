"""Self-tests for the suite-wide guards installed by tests/conftest.py.

These pin the guards themselves (UPG-TEST-LIVE-DAEMON-PORT-CONTAMINATION,
UPG-TEST-REAL-EMBEDDER-DOWNLOAD-GUARD): if a refactor silently disconnects
a guard from the seam it protects, these tests must fail — otherwise the
guards rot into security theater while the suite goes back to probing live
daemons and reaching for huggingface.co.
"""
from __future__ import annotations

import concurrent.futures
import socket

import pytest

import tests.conftest as cf
from tests.conftest import (
    RealModelDownloadBlocked,
    _BlockedConnectionAttempt,
    _current_loopback_violations,
    _inspect_connect_address,
    _summarize_violations,
    allow_loopback_port,
)


# ---------------------------------------------------------------------------
# Socket guard — address classification (pure logic)
# ---------------------------------------------------------------------------

class TestConnectAddressClassification:
    def test_unregistered_loopback_ip_classified_as_violation(self):
        assert _inspect_connect_address(("127.0.0.1", 8765))[0] == "loopback"

    def test_localhost_name_classified_as_loopback(self):
        assert _inspect_connect_address(("localhost", 8765))[0] == "loopback"

    def test_ipv6_loopback_classified_as_loopback(self):
        assert _inspect_connect_address(("::1", 8765, 0, 0))[0] == "loopback"
        assert _inspect_connect_address(("0:0:0:0:0:0:0:1", 8765))[0] == "loopback"

    def test_whole_ipv4_loopback_range_covered(self):
        # 127.0.0.0/8, not just .0.1 — probes sometimes use 127.0.0.2-style
        # addresses to dodge naive equality checks.
        assert _inspect_connect_address(("127.7.7.7", 1234))[0] == "loopback"

    def test_registered_port_is_allowed(self):
        with allow_loopback_port(45999):
            assert _inspect_connect_address(("127.0.0.1", 45999))[0] == "allowed"
        # Registration is scoped to the context manager: after exit the same
        # address is back to being a violation candidate.
        assert _inspect_connect_address(("127.0.0.1", 45999))[0] == "loopback"

    def test_remote_address_is_not_a_violation_candidate(self):
        kind, host, port = _inspect_connect_address(("example.com", 443))
        assert (kind, host, port) == ("remote", "example.com", 443)

    def test_non_ip_targets_pass_through_untouched(self):
        # AF_UNIX path / malformed shapes → None = "not our business".
        assert _inspect_connect_address("/tmp/some.sock") is None
        assert _inspect_connect_address(("127.0.0.1",)) is None
        assert _inspect_connect_address(("127.0.0.1", "not-a-port")) is None


# ---------------------------------------------------------------------------
# Socket guard — live behavior against real sockets
# ---------------------------------------------------------------------------

class TestSocketGuardLiveBehavior:
    def test_remote_connect_attempt_is_refused_and_not_recorded(self):
        """Any egress attempt dies immediately (the suite is network-free),
        but remote attempts are NOT violations: third-party telemetry noise
        must not fail unrelated tests. If the guard were absent, this connect
        would either succeed (real network) or hang until timeout on a
        filtered network — exactly the nondeterminism the guard removes."""
        with pytest.raises(_BlockedConnectionAttempt):
            socket.create_connection(("203.0.113.1", 6443), timeout=0.5)
        # ConnectionError-compatible so product broad-excepts see an ordinary
        # "unreachable" verdict rather than a novel exception type.
        assert issubclass(_BlockedConnectionAttempt, ConnectionError)
        assert _current_loopback_violations() == []

    def test_allowlisted_local_server_is_reachable_end_to_end(self):
        """The escape hatch works against a REAL listener: a test-owned stub
        server on an ephemeral port accepts connections while registered."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            with allow_loopback_port(port):
                client = socket.create_connection(("127.0.0.1", port), timeout=2)
                client.close()
            assert _current_loopback_violations() == []
        finally:
            srv.close()

    def test_unallowlisted_loopback_connect_raises_and_records(self):
        """The core contamination scenario: a probe toward a port nobody
        registered (e.g. the user's live daemon) raises immediately AND lands
        in the violation ledger so the owning test fails post-hoc."""
        with pytest.raises(_BlockedConnectionAttempt, match="socket guard"):
            socket.create_connection(("127.0.0.1", 59877), timeout=0.5)
        assert ("127.0.0.1", 59877) in _current_loopback_violations()
        # This test DELIBERATELY produced a violation; clear the ledger so
        # the autouse owning-test failure (which exists for accidental
        # violations) doesn't fire on the guard's own exercise of itself.
        with cf._violations_lock:
            cf._loopback_violations.clear()

    def test_recorded_violation_message_names_target_and_both_remediations(self):
        message = _summarize_violations([("127.0.0.1", 8765), ("localhost", 8766)])
        assert message is not None
        assert "127.0.0.1:8765" in message and "localhost:8766" in message
        assert "_is_server_alive" in message       # stub-the-probe remedy
        assert "allow_loopback_port" in message    # own-a-stub-server remedy
        assert _summarize_violations([]) is None


# ---------------------------------------------------------------------------
# Real-model download guard
# ---------------------------------------------------------------------------

def _test_builder(local_files_only):
    return f"built(local_files_only={local_files_only})"


class _RealHFClassStandin:
    """Mimics what the guard detects in a product builder's closure: an
    object named like sentence_transformers' class AND claiming that
    module — without importing torch."""


_RealHFClassStandin.__name__ = "SentenceTransformer"
_RealHFClassStandin.__module__ = "sentence_transformers"


def _make_product_builder():
    """The capture must be a real CLOSURE CELL, not a module global.

    `LocalEmbedProvider.__init__` from-imports SentenceTransformer as a
    function local and its builder lambda closes over that local
    (agent/indexer/_types.py), which is why the guard inspects
    `build_fn.__closure__`. A stand-in that merely READS a module-level name
    has `__closure__ is None` and would exempt itself from the very rule it
    claims to exercise — the test would pass for the wrong reason if the
    guard were later weakened to inspect globals instead.
    """
    st = _RealHFClassStandin  # captured, exactly like the product lambda's ST

    def _builder(local_files_only):
        return f"would construct {st!r} with local_files_only={local_files_only}"

    # Make this builder LOOK product-defined (the guard keys off the builder's
    # __module__) without importing anything heavy.
    _builder.__module__ = "agent.indexer._types"
    return _builder


_product_builder = _make_product_builder()


class TestRealModelDownloadGuard:
    def test_product_builder_for_uncached_model_is_refused(self, tmp_path):
        """The CI incident's exact shape: product code asks the loader for a
        model that isn't in the (session-isolated) cache → refuse BEFORE any
        construction. If the guard were absent, the loader would call
        build_fn(local_files_only=False) and sentence-transformers would head
        for huggingface.co."""
        import agent.model_cache as mc

        with pytest.raises(RealModelDownloadBlocked, match="unit suite must never download"):
            mc.load_with_offline_preference(
                _product_builder, "org/model-not-in-cache", str(tmp_path),
            )

    def test_test_defined_builder_delegates_to_real_loader_even_on_miss(self, tmp_path):
        """Tests exercising the LOADER itself (tests/test_model_cache.py) pass
        their own fake builders — they must keep working untouched, including
        on the cache-miss branch."""
        import agent.model_cache as mc

        out = mc.load_with_offline_preference(
            _test_builder, "org/model-not-in-cache", str(tmp_path),
        )
        assert out == "built(local_files_only=False)"

    def test_product_builder_with_cached_model_delegates_offline_branch(self, tmp_path, monkeypatch):
        """A cached model never downloads, so the guard steps aside and the
        loader's offline branch runs (build_fn(True))."""
        import agent.model_cache as mc

        monkeypatch.setattr(mc, "is_model_cached", lambda name, cache_dir: True)
        out = mc.load_with_offline_preference(
            _product_builder, "org/cached-model", str(tmp_path),
        )
        # `_product_builder` reports the class it WOULD have constructed; the
        # only thing under test here is that the loader's offline branch ran
        # it (local_files_only=True) instead of the guard refusing it.
        assert out.endswith("local_files_only=True"), out

    def test_builder_without_product_module_is_treated_as_exempt(self, tmp_path):
        """Defensive default: an exotic callable whose __module__ isn't under
        agent/ (here: a plain object, resolving to its test-module class
        attr) is let through at classification — fail-open there, fail-loud
        at the hub backstop — rather than crashing the classifier itself."""
        import agent.model_cache as mc

        class _CallableBuilder:
            def __call__(self, local_files_only):
                return f"built({local_files_only})"

        out = mc.load_with_offline_preference(
            _CallableBuilder(), "org/model-not-in-cache", str(tmp_path),
        )
        assert out == "built(False)"

    def test_huggingface_hub_backstop_refuses_downloads_directly(self):
        """Paths that skip our loader (direct SentenceTransformer-style
        construction) still die at the hub boundary."""
        import huggingface_hub

        with pytest.raises(RealModelDownloadBlocked):
            huggingface_hub.snapshot_download("org/model-not-in-cache")

    def test_unstubbed_local_embed_provider_construction_fails_loudly(self):
        """End-to-end seam proof: LocalEmbedProvider.__init__ resolves
        load_with_offline_preference via a lazy from-import, so the session
        guard MUST be what that import finds at call time. If someone renames
        the lazy import or hoists it to module level, this test fails while
        the wrapper-level tests above stay green — that gap is exactly what
        this pins. (First call pays the torch/transformers import cost once
        per session; later constructions reuse it.)"""
        from agent.indexer._types import LocalEmbedProvider

        with pytest.raises(
            RealModelDownloadBlocked, match="org/not-a-real-model-for-tests",
        ):
            LocalEmbedProvider("org/not-a-real-model-for-tests")


# ---------------------------------------------------------------------------
# Drain-barrier helper sanity (full semantics pinned in
# tests/test_indexer_searcher.py::TestPurposePassDeferral)
# ---------------------------------------------------------------------------

class TestDrainBarrierEdgeCases:
    def test_barrier_returns_immediately_without_executor(self):
        from tests.conftest import wait_for_deferred_purpose_pass

        class _IdxWithoutExecutor:
            purpose_vectors_pending = 0

        wait_for_deferred_purpose_pass(_IdxWithoutExecutor(), timeout=0.1)

    def test_barrier_works_against_a_real_idle_single_worker_executor(self):
        from tests.conftest import wait_for_deferred_purpose_pass

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vectr-purpose",
        )
        try:

            class _Idx:
                _purpose_executor = executor
                purpose_vectors_pending = 0

            wait_for_deferred_purpose_pass(_Idx(), timeout=5)
        finally:
            executor.shutdown(wait=True)
