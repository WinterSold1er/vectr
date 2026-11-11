"""Tests for the PreCompact boundary-preservation surface:
`VectrService._boundary_precompact_text` / `.boundary_precompact()`,
`GET /v1/boundary/precompact`, and the two new
`agent/config.yaml` `episodes.boundary_precompact_*` keys.

vectr never decides what is "worth keeping" here (zero-inference core): the
rendered text is a fixed, arc-independent nudge plus a single deterministic
integer-count sentence — no content classification, no query-conditional
branching, nothing gated on what the session actually discussed. The
summarizer (the editor's own LLM) does all the judgment; this surface only
tells it what NOT to let compaction throw away.
"""
from __future__ import annotations

import time as time_module
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent.trigger_engine import token_estimate
from tests.conftest import _DummyEmbedProvider


def _make_real_service(tmp_path, monkeypatch):
    """Mirrors `_make_real_service` in tests/test_arc_distill.py — a real
    VectrService, dummy embedder, own workspace root."""
    from agent import indexer as idx_module

    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
    with patch("integrations.vscode_bridge.configure_all"), \
         patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
         patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
        from app.service import VectrService
        svc = VectrService(workspace_root=str(tmp_path))
    return svc


@pytest.fixture
def real_service(tmp_path, monkeypatch):
    """Function-scoped real VectrService, dedicated to this file (same
    isolation rationale as test_arc_distill.py's `real_client`)."""
    return _make_real_service(tmp_path, monkeypatch)


@pytest.fixture
def real_client(tmp_path, monkeypatch):
    """Function-scoped TestClient over a REAL VectrService."""
    svc = _make_real_service(tmp_path, monkeypatch)
    from api import app
    prior = getattr(app.state, "service", None)
    with patch("app.service.VectrService", return_value=svc):
        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.service = svc
            try:
                yield c, svc
            finally:
                app.state.service = prior


def _insert_pending_arc(svc, ts: float | None = None) -> int:
    """Write one pending arc directly through `EpisodeStore` — the same
    shape `_insert_arc` in tests/test_arc_distill.py builds — without going
    through the full episode-ingestion detection pipeline, since this file
    only needs `count_arcs_pending_distill()` to be positive."""
    ts = time_module.time() if ts is None else ts
    store = svc._episode_store
    ws = svc._workspace_root
    fail_id = store.insert(
        ws, session_id="s1", ts=ts, cwd="/repo", tool="bash", cmd_raw="pytest -k a",
        verb="pytest", flags=["-k"], args=["a"], rc=1, termination="normal",
        outcome="failure", stdout_digest="", stderr_digest="", markers_matched=[],
        env_delta_names=[], file_path=None, max_rows=1000, ttl_days=30,
    )
    success_id = store.insert(
        ws, session_id="s1", ts=ts + 1, cwd="/repo", tool="bash", cmd_raw="pytest -k b",
        verb="pytest", flags=["-k"], args=["b"], rc=0, termination="normal",
        outcome="success", stdout_digest="", stderr_digest="", markers_matched=[],
        env_delta_names=[], file_path=None, max_rows=1000, ttl_days=30,
    )
    arc_id = store.insert_arc(
        ws, session_id="s1", cwd="/repo", ts=ts, confidence="normal",
        mutation_diff={"flag": (("a",), ("b",))},
        failure_episode_ids=[fail_id], success_episode_id=success_id,
    )
    store.mark_episode_arc(fail_id, arc_id)
    store.mark_episode_arc(success_id, arc_id)
    return arc_id


# ---------------------------------------------------------------------------
# VectrService._boundary_precompact_text / .boundary_precompact()
# ---------------------------------------------------------------------------


class TestBoundaryPrecompactRenderer:
    def test_zero_arcs_returns_base_text_only(self, real_service):
        from app.service import _BOUNDARY_PRECOMPACT_BASE_TEXT

        assert real_service.count_arcs_pending_distill() == 0
        assert real_service._boundary_precompact_text() == _BOUNDARY_PRECOMPACT_BASE_TEXT

    def test_positive_arcs_appends_exactly_one_count_sentence(self, real_service):
        from app.service import _BOUNDARY_PRECOMPACT_BASE_TEXT

        _insert_pending_arc(real_service)
        text = real_service._boundary_precompact_text()

        assert text.startswith(_BOUNDARY_PRECOMPACT_BASE_TEXT)
        assert text.count("pending distillation") == 1
        assert "1 command-discovery arc(s) recorded this session are still " \
               "pending distillation; keep the reasoning about what they mean." in text

    def test_arc_sentence_count_matches_pending_count_exactly(self, real_service):
        """Deterministic integer count, not a rounded/approximate figure —
        no content classification of the arcs themselves."""
        _insert_pending_arc(real_service, ts=time_module.time())
        _insert_pending_arc(real_service, ts=time_module.time() - 5)
        text = real_service._boundary_precompact_text()
        assert "2 command-discovery arc(s)" in text

    def test_determinism_byte_identical_across_calls(self, real_service):
        _insert_pending_arc(real_service)
        first = real_service._boundary_precompact_text()
        second = real_service._boundary_precompact_text()
        assert first == second

    def test_base_text_always_fits_token_cap(self):
        from agent.config import BOUNDARY_PRECOMPACT_TOKEN_CAP
        from app.service import _BOUNDARY_PRECOMPACT_BASE_TEXT

        assert token_estimate(_BOUNDARY_PRECOMPACT_BASE_TEXT) <= BOUNDARY_PRECOMPACT_TOKEN_CAP

    def test_tiny_cap_drops_arc_sentence_keeps_base_text(self, real_service, monkeypatch):
        """When the combined (base + arc sentence) text would not fit the
        configured cap, the arc sentence is dropped entirely — the base
        text alone is returned unmodified, never truncated mid-sentence.

        UPG-L3-CONFIG-READ-TIME: the renderer reads the cap at request time
        through `agent.config`, so that module attribute (not any importer's
        by-value binding) is the override point."""
        from app.service import _BOUNDARY_PRECOMPACT_BASE_TEXT

        _insert_pending_arc(real_service)
        base_tokens = token_estimate(_BOUNDARY_PRECOMPACT_BASE_TEXT)
        monkeypatch.setattr("agent.config.BOUNDARY_PRECOMPACT_TOKEN_CAP", base_tokens)

        text = real_service._boundary_precompact_text()
        assert text == _BOUNDARY_PRECOMPACT_BASE_TEXT
        assert "pending distillation" not in text

    def test_disabled_config_returns_empty_string_even_with_pending_arcs(self, real_service, monkeypatch):
        _insert_pending_arc(real_service)
        monkeypatch.setattr("agent.config.BOUNDARY_PRECOMPACT_ENABLED", False)
        assert real_service._boundary_precompact_text() == ""

    def test_constants_are_not_rebound_into_app_service(self):
        """UPG-L3-CONFIG-READ-TIME tripwire: app.service must NOT re-bind
        BOUNDARY_PRECOMPACT_* as its own module attributes. If these names
        reappear there, the monkeypatch.setattr("agent.config.…") overrides
        above silently stop reaching the renderer while still passing — the
        exact dual-patch-point trap this reconciliation removed. Reverting
        the service-side change must fail THIS test loudly, not just the
        ones above quietly testing nothing."""
        import app.service
        assert not hasattr(app.service, "BOUNDARY_PRECOMPACT_ENABLED")
        assert not hasattr(app.service, "BOUNDARY_PRECOMPACT_TOKEN_CAP")

    def test_no_note_content_or_ids_in_text(self, real_service):
        """Notes already survive compaction via the separate SessionStart
        `compact` re-injection path (UPG-9.4) — restating note content or
        ids here would be a double-dip, not additive information."""
        note_id = real_service.remember(
            "the workspace lock is PID-scoped and released on scope exit",
            kind="finding",
        )
        _insert_pending_arc(real_service)
        text = real_service._boundary_precompact_text()

        assert "workspace lock" not in text
        assert "PID-scoped" not in text
        assert f"#{note_id}" not in text
        # The only integer this text may ever contain is the pending-arc
        # count sentence's own count — never a note id. Checking the bare
        # numeral would be a false positive here (a small note id can
        # coincidentally equal the arc count), so assert on the id's
        # distinguishing reference form instead.
        assert f"note {note_id}" not in text
        assert f"note_id={note_id}" not in text

    def test_boundary_precompact_dict_shape(self, real_service):
        _insert_pending_arc(real_service)
        result = real_service.boundary_precompact()
        assert set(result.keys()) == {"text", "arcs_pending"}
        assert result["arcs_pending"] == 1
        assert result["text"] == real_service._boundary_precompact_text()

    def test_boundary_precompact_reads_the_count_once(self, real_service):
        """One response = one store snapshot. The renderer must be handed the
        count the response reports, not read its own: two reads could straddle
        a newly-formed arc and publish a `text` naming a different number than
        the `arcs_pending` field beside it."""
        _insert_pending_arc(real_service)
        calls = {"n": 0}
        real_count = real_service.count_arcs_pending_distill

        def counting():
            calls["n"] += 1
            return real_count()

        real_service.count_arcs_pending_distill = counting
        try:
            result = real_service.boundary_precompact()
        finally:
            real_service.count_arcs_pending_distill = real_count

        assert calls["n"] == 1, f"expected a single store read, got {calls['n']}"
        assert f"{result['arcs_pending']} command-discovery arc(s)" in result["text"]

    def test_renderer_honors_an_explicitly_passed_count(self, real_service):
        """The passed count wins over the store, so the caller's snapshot is
        what gets rendered."""
        _insert_pending_arc(real_service)  # store says 1
        text = real_service._boundary_precompact_text(arcs_pending=7)
        assert "7 command-discovery arc(s)" in text
        assert "1 command-discovery arc(s)" not in text


# ---------------------------------------------------------------------------
# GET /v1/boundary/precompact
# ---------------------------------------------------------------------------


class TestBoundaryPrecompactRestRoute:
    def test_returns_200_with_declared_fields_zero_arcs(self, real_client):
        client, svc = real_client
        resp = client.get("/v1/boundary/precompact")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"text", "arcs_pending"}
        assert data["arcs_pending"] == 0
        assert "pending distillation" not in data["text"]
        assert data["text"] == svc._boundary_precompact_text()

    def test_returns_arc_sentence_and_count_when_arcs_pending(self, real_client):
        client, svc = real_client
        _insert_pending_arc(svc)
        resp = client.get("/v1/boundary/precompact")
        assert resp.status_code == 200
        data = resp.json()
        assert data["arcs_pending"] == 1
        assert "1 command-discovery arc(s)" in data["text"]

    def test_disabled_config_returns_empty_text_still_200(self, real_client, monkeypatch):
        client, svc = real_client
        # UPG-L3-CONFIG-READ-TIME: overridden at the config module — honored
        # end-to-end through the service's request-time read.
        monkeypatch.setattr("agent.config.BOUNDARY_PRECOMPACT_ENABLED", False)
        resp = client.get("/v1/boundary/precompact")
        assert resp.status_code == 200
        assert resp.json()["text"] == ""

    def test_search_only_mode_returns_503(self):
        """Parity with GET /v1/arcs — memory-facing REST routes are gated
        at the route level on `svc.search_only`, before the service method
        is ever called."""
        from api import app
        from tests.conftest import _base_mock_service

        svc = _base_mock_service()
        svc.search_only = True

        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=False) as c:
                app.state.service = svc
                resp = c.get("/v1/boundary/precompact")
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "search_only_mode"
        svc.boundary_precompact.assert_not_called()


# ---------------------------------------------------------------------------
# agent/config.yaml episodes.boundary_precompact_* keys
# ---------------------------------------------------------------------------


class TestBoundaryPrecompactConfig:
    def test_config_values_loaded_from_yaml_not_hardcoded_defaults(self):
        """A missing key raises KeyError at import per this repo's
        config-access rule (agent/config.py uses direct subscript access,
        never `.get()` with a default) — a successful import here already
        proves the keys exist in agent/config.yaml; this also pins the
        shipped values so an accidental edit is caught."""
        from agent.config import (
            BOUNDARY_PRECOMPACT_ENABLED,
            BOUNDARY_PRECOMPACT_TOKEN_CAP,
        )
        assert isinstance(BOUNDARY_PRECOMPACT_ENABLED, bool)
        assert isinstance(BOUNDARY_PRECOMPACT_TOKEN_CAP, int)
        assert BOUNDARY_PRECOMPACT_ENABLED is True
        assert BOUNDARY_PRECOMPACT_TOKEN_CAP == 200
