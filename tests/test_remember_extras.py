"""Tests for VectrService.remember_with_extras() (app/service.py) — the
additive write-time offers surface (related notes + revoked-related
deterrent + proxy-anchor suggestions) layered on top of the existing
remember() write.

remember_with_extras() is a NEW method: remember() itself is untouched
(same signature, same int return, same 495+ existing call sites). These
tests exercise the gating matrix documented on remember_with_extras()'s own
docstring:
  - related notes: gated on MEMORY_WRITE_RELATED_ENABLED only.
  - revoked-related deterrent: gated on MEMORY_WRITE_RELATED_REVOKED_ENABLED
    only (UPG-RELATED-REVOKED-DETERRENT).
  - proxy-anchor suggestions: gated on MEMORY_WRITE_PROXY_SUGGEST_ENABLED
    AND kind == "operational" AND no anchors passed on this call.
  - a failure anywhere in the extras computation must never break the
    write — note_id must still come back valid, all three lists degrade
    to [].
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.service import RememberOutcome, VectrService
from tests.conftest import _DummyEmbedProvider


def _make_service(tmp_path, monkeypatch) -> VectrService:
    """A real VectrService (real WorkingContextStore, real notes-collection
    embedder) backed by the deterministic zero-download dummy embed
    provider — same construction shape as test_arc_distill.py's
    `_make_real_service`. `_DummyEmbedProvider` seeds each vector on
    `content[:80]`, so two notes sharing that prefix embed IDENTICALLY
    (cosine similarity 1.0) — a real, non-mocked way to force a related-note
    hit without hand-rolling ChromaDB internals."""
    from agent import indexer as idx_module
    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider())
    with patch("integrations.vscode_bridge.configure_all"), \
         patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
         patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
        return VectrService(workspace_root=str(tmp_path))


class TestRememberWithExtrasBasicContract:
    def test_returns_remember_outcome_with_valid_note_id(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("a plain finding, nothing else stored yet")
        assert isinstance(outcome, RememberOutcome)
        assert isinstance(outcome.note_id, int)
        assert svc.get_note(outcome.note_id) is not None

    def test_note_content_matches_a_plain_remember_call(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("second note body")
        note = svc.get_note(outcome.note_id)
        assert note.content == "second note body"

    def test_value_error_from_underlying_write_propagates(self, tmp_path, monkeypatch) -> None:
        """remember_with_extras performs the SAME write remember() does —
        an invalid `supersedes` target must raise exactly as remember()
        itself does, never swallowed by the extras try/except (that guard
        wraps only the offer computation, which runs after the write)."""
        svc = _make_service(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            svc.remember_with_extras("a note", supersedes=999999)


class TestRememberWithExtrasRelatedGating:
    def test_related_populated_when_near_duplicate_exists(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        content = "shared exact content for near-duplicate similarity"
        base_id = svc.remember(content)
        outcome = svc.remember_with_extras(content)
        assert any(r["note_id"] == base_id for r in outcome.related)

    def test_related_empty_when_no_candidate_exists(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("solo unique content, nothing else stored")
        assert outcome.related == []

    def test_related_empty_when_disabled(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        content = "shared exact content for near-duplicate similarity"
        svc.remember(content)
        monkeypatch.setattr("app.service.MEMORY_WRITE_RELATED_ENABLED", False)
        outcome = svc.remember_with_extras(content)
        assert outcome.related == []


class TestRememberWithExtrasRevokedRelatedGating:
    def test_revoked_related_populated_when_revoked_near_duplicate_exists(
        self, tmp_path, monkeypatch,
    ) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        content = "shared exact content for revoked near-duplicate similarity"
        base_id = svc.remember(content)
        assert svc.revoke_note(base_id, reason="turned out to be wrong") is True

        outcome = svc.remember_with_extras(content)

        assert any(r["note_id"] == base_id for r in outcome.revoked_related)
        match = next(r for r in outcome.revoked_related if r["note_id"] == base_id)
        assert match["reason"] == "turned out to be wrong"
        assert isinstance(match["revoked_date"], str) and match["revoked_date"]

    def test_revoked_related_empty_when_the_near_duplicate_is_still_active(
        self, tmp_path, monkeypatch,
    ) -> None:
        """An active (non-revoked) near-duplicate must never appear in
        `revoked_related` — that is exactly what `related` (above) is for.
        The two lists answer different questions and must never overlap on
        an active candidate."""
        svc = _make_service(tmp_path, monkeypatch)
        content = "shared exact content for an active near-duplicate"
        svc.remember(content)

        outcome = svc.remember_with_extras(content)

        assert outcome.revoked_related == []

    def test_revoked_related_empty_when_no_candidate_exists(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("solo unique content, nothing else stored")
        assert outcome.revoked_related == []

    def test_revoked_related_empty_when_disabled(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        content = "shared exact content for revoked near-duplicate, disabled case"
        base_id = svc.remember(content)
        svc.revoke_note(base_id, reason="wrong")
        monkeypatch.setattr("app.service.MEMORY_WRITE_RELATED_REVOKED_ENABLED", False)
        outcome = svc.remember_with_extras(content)
        assert outcome.revoked_related == []


class TestRememberWithExtrasProxyGating:
    def test_populated_for_operational_kind_with_no_anchors(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("deploy note", kind="operational")
        assert "Dockerfile" in outcome.proxy_anchor_suggestions

    def test_empty_for_non_operational_kind(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("a finding", kind="finding")
        assert outcome.proxy_anchor_suggestions == []

    def test_empty_when_caller_already_passed_anchors(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras(
            "deploy note", kind="operational", anchors=["Dockerfile"],
        )
        assert outcome.proxy_anchor_suggestions == []

    def test_empty_when_anchors_is_an_empty_list(self, tmp_path, monkeypatch) -> None:
        """An explicit `anchors=[]` is treated the same as omitted — the
        caller declared no anchors either way — so the suggestion still
        fires (the gate is `not anchors`, matching the rest of the memory
        layer's None-or-empty-means-omitted convention)."""
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("deploy note", kind="operational", anchors=[])
        assert "Dockerfile" in outcome.proxy_anchor_suggestions

    def test_empty_when_disabled(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
        svc = _make_service(tmp_path, monkeypatch)
        monkeypatch.setattr("app.service.MEMORY_WRITE_PROXY_SUGGEST_ENABLED", False)
        outcome = svc.remember_with_extras("deploy note", kind="operational")
        assert outcome.proxy_anchor_suggestions == []

    def test_empty_when_no_proxy_files_present(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)
        outcome = svc.remember_with_extras("deploy note", kind="operational")
        assert outcome.proxy_anchor_suggestions == []


class TestRememberWithExtrasFailSafe:
    def test_write_succeeds_even_when_related_lookup_raises(self, tmp_path, monkeypatch) -> None:
        svc = _make_service(tmp_path, monkeypatch)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated related-notes failure")

        monkeypatch.setattr(svc._context_store, "related_active_notes", _boom)
        outcome = svc.remember_with_extras("a note that must still be written")

        assert isinstance(outcome.note_id, int)
        assert svc.get_note(outcome.note_id) is not None
        assert outcome.related == []
        assert outcome.proxy_anchor_suggestions == []

    def test_write_succeeds_even_when_revoked_related_lookup_raises(
        self, tmp_path, monkeypatch,
    ) -> None:
        svc = _make_service(tmp_path, monkeypatch)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated revoked-related-notes failure")

        monkeypatch.setattr(svc._context_store, "revoked_related_notes", _boom)
        outcome = svc.remember_with_extras("a note that must still be written")

        assert isinstance(outcome.note_id, int)
        assert svc.get_note(outcome.note_id) is not None
        assert outcome.revoked_related == []
        assert outcome.proxy_anchor_suggestions == []

    def test_write_succeeds_even_when_proxy_suggestion_raises(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
        svc = _make_service(tmp_path, monkeypatch)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated proxy-anchor failure")

        monkeypatch.setattr("app.service.suggest_proxy_anchors", _boom)
        outcome = svc.remember_with_extras("deploy note", kind="operational")

        assert isinstance(outcome.note_id, int)
        assert svc.get_note(outcome.note_id) is not None
        assert outcome.proxy_anchor_suggestions == []

    def test_one_offer_raising_does_not_leave_the_other_partially_computed(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The extras computation degrades as a WHOLE on any failure — a
        related-notes exception must not leave an already-computed proxy
        suggestion list behind (or vice versa); both come back empty."""
        (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
        svc = _make_service(tmp_path, monkeypatch)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated related-notes failure")

        monkeypatch.setattr(svc._context_store, "related_active_notes", _boom)
        outcome = svc.remember_with_extras("deploy note", kind="operational")

        assert outcome.related == []
        assert outcome.proxy_anchor_suggestions == []
