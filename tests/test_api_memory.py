"""
Tests for REST API memory routes: POST /v1/remember, POST /v1/recall,
POST /v1/snapshot, GET /v1/evict-hint, GET /v1/snapshot/list, GET /v1/resume.

Two fixture layers:
  client              — mocked VectrService, tests HTTP shape/status codes
  client_real_memory  — real WorkingContextStore, tests cross-request persistence
    (the exact flow that failed in the POC benchmark: Phase 1 remember → Phase 2 recall)
"""
from __future__ import annotations

import pytest

from app.service import VectrService
from tests._seam import assert_seam_call


# ---------------------------------------------------------------------------
# POST /v1/remember
# ---------------------------------------------------------------------------

class TestRememberRoute:
    def test_remember_returns_note_id(self, client) -> None:
        resp = client.post("/v1/remember", json={"content": "Signal.send at dispatcher.py:220"})
        assert resp.status_code == 200
        data = resp.json()
        assert "note_id" in data
        assert "message" in data

    def test_remember_with_tags_and_priority(self, client) -> None:
        resp = client.post("/v1/remember", json={
            "content": "BaseHandler.load_middleware at django/core/handlers/base.py",
            "tags": ["middleware", "lifecycle"],
            "priority": "high",
        })
        assert resp.status_code == 200
        assert resp.json()["note_id"] is not None

    def test_remember_missing_content_returns_422(self, client) -> None:
        resp = client.post("/v1/remember", json={})
        assert resp.status_code == 422

    def test_remember_message_suggests_eviction(self, client) -> None:
        resp = client.post("/v1/remember", json={"content": "some note"})
        msg = resp.json()["message"]
        assert "drop" in msg.lower() or "evict" in msg.lower() or "stored" in msg.lower()

    def test_remember_processing_ms_present(self, client) -> None:
        resp = client.post("/v1/remember", json={"content": "timing test"})
        assert "processing_ms" in resp.json()

    def test_remember_accepts_agent_field(self, client) -> None:
        """UPG-SUBAGENT-MEMORY: optional agent identifier is accepted, not rejected."""
        resp = client.post("/v1/remember", json={"content": "subagent finding", "agent": "coder-2"})
        assert resp.status_code == 200
        assert resp.json()["note_id"] is not None

    def test_remember_agent_defaults_to_empty(self, client) -> None:
        resp = client.post("/v1/remember", json={"content": "no agent set"})
        assert resp.status_code == 200

    # -- TRIGGER-ENGINE wave 1 (bm2-design-skeleton.md §1/§2/§5) --------------

    def test_remember_accepts_trigger_engine_params(self, client) -> None:
        """All new params are additive/optional and round-trip through the
        mocked service without changing the pre-existing response shape."""
        resp = client.post("/v1/remember", json={
            "content": "a gotcha about auth.py",
            "kind": "gotcha",
            "triggers": [{"path": "src/auth.py", "event": "pre-edit"}],
            "provenance": "auto",
            "scope": "repo",
            "anchors": ["src/auth.py"],
        })
        assert resp.status_code == 200
        assert resp.json()["note_id"] is not None

    def test_remember_omitting_trigger_engine_params_is_unchanged(self, client) -> None:
        """No new params present reproduces exactly the pre-wave-1 call."""
        resp = client.post("/v1/remember", json={"content": "plain note, no trigger params"})
        assert resp.status_code == 200

    def test_remember_invalid_provenance_rejected_at_rest_boundary(self, client) -> None:
        # 'human' is not settable via REST (only via /v1/promote).
        resp = client.post("/v1/remember", json={"content": "x", "provenance": "human"})
        assert resp.status_code == 422

    def test_remember_invalid_scope_rejected_at_rest_boundary(self, client) -> None:
        resp = client.post("/v1/remember", json={"content": "x", "scope": "not-a-real-scope"})
        assert resp.status_code == 422

    # -- UPG-TRIGGER-SCOPE-KIND-DEFAULTS --------------------------------------

    def test_remember_omitted_scope_resolves_to_kind_default_via_rest(self, client_real_memory, tmp_path) -> None:
        """A `scope`-omitting REST call (the JSON body never includes the
        key) reaches the store as None, not the string "workspace" — so a
        kind="gotcha" note's OMITTED scope resolves to "repo" end-to-end
        through the real route/model/service/store stack, not just
        unit-level."""
        import sqlite3
        resp = client_real_memory.post("/v1/remember", json={
            "content": "a gotcha about auth.py", "kind": "gotcha",
        })
        assert resp.status_code == 200
        note_id = resp.json()["note_id"]
        with sqlite3.connect(str(tmp_path / "working_context.sqlite")) as conn:
            row = conn.execute("SELECT scope FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        assert row[0] == "repo"

    def test_remember_explicit_workspace_scope_overrides_kind_default_via_rest(self, client_real_memory, tmp_path) -> None:
        """An explicitly passed scope="workspace" in the REST body always
        wins verbatim, even for a kind (gotcha) whose omitted default would
        otherwise be "repo"."""
        import sqlite3
        resp = client_real_memory.post("/v1/remember", json={
            "content": "a gotcha about auth.py", "kind": "gotcha", "scope": "workspace",
        })
        assert resp.status_code == 200
        note_id = resp.json()["note_id"]
        with sqlite3.connect(str(tmp_path / "working_context.sqlite")) as conn:
            row = conn.execute("SELECT scope FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        assert row[0] == "workspace"

    def test_remember_malformed_triggers_returns_422(self, client_real_memory) -> None:
        """A malformed trigger reaches the store's ValueError and surfaces as
        a 422 (caller input error), not a 500."""
        resp = client_real_memory.post("/v1/remember", json={
            "content": "note", "triggers": [{"not_before": 1.0}],
        })
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "invalid_memory_object"

    def test_remember_auto_provenance_rejected_on_directive_via_rest(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/remember", json={
            "content": "an unreviewed standing rule", "kind": "directive", "provenance": "auto",
        })
        assert resp.status_code == 422

    def test_remember_supersedes_round_trips_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        old = client.post("/v1/remember", json={"content": "old finding"}).json()["note_id"]
        new = client.post("/v1/remember", json={"content": "corrected finding", "supersedes": old}).json()["note_id"]
        assert new != old
        active = client.post("/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "corrected finding" in active
        assert "old finding" not in active

    def test_remember_contradicts_revokes_the_target_via_rest(self, client_real_memory) -> None:
        """UPG-MEMORY-STATE-MACHINE §4.2: unlike supersedes, the target
        stays a live recall candidate — rendered as anti-memory, not
        excluded."""
        client = client_real_memory
        old = client.post("/v1/remember", json={"content": "the API returns a list"}).json()["note_id"]
        new = client.post(
            "/v1/remember",
            json={"content": "the API actually returns a dict", "contradicts": old},
        ).json()["note_id"]
        assert new != old
        full = client.post("/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "Previously believed" in full
        assert "the API actually returns a dict" in full

    def test_remember_contradicts_nonexistent_note_returns_422(self, client_real_memory) -> None:
        resp = client_real_memory.post(
            "/v1/remember", json={"content": "a correction", "contradicts": 999999}
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "invalid_memory_object"

    # -- UPG-RELATED-REVOKED-DETERRENT ---------------------------------------

    def test_remember_response_carries_revoked_related_field(self, client) -> None:
        """`/v1/remember`'s response model exposes `revoked_related`
        end-to-end from `RememberOutcome.revoked_related` (app/routes.py),
        distinct from and alongside `related` — a REST-layer test, not just
        the MCP dispatch/service-layer coverage in test_mcp_server.py and
        test_remember_extras.py (MCP-green is not REST-green)."""
        from api import app
        from app.service import RememberOutcome
        app.state.service.remember_with_extras.return_value = RememberOutcome(
            note_id=42,
            related=[],
            revoked_related=[{
                "note_id": 7, "title": "an older note that was revoked",
                "similarity": 0.86, "revoked_date": "2026-08-01",
                "reason": "turned out to be wrong",
            }],
            proxy_anchor_suggestions=[],
        )
        resp = client.post("/v1/remember", json={"content": "a new note"})
        assert resp.status_code == 200
        revoked = resp.json()["revoked_related"]
        assert len(revoked) == 1
        assert revoked[0] == {
            "note_id": 7, "title": "an older note that was revoked",
            "similarity": 0.86, "revoked_date": "2026-08-01",
            "reason": "turned out to be wrong",
        }

    def test_remember_response_revoked_related_defaults_to_empty_list(self, client) -> None:
        resp = client.post("/v1/remember", json={"content": "a plain note"})
        assert resp.status_code == 200
        assert resp.json()["revoked_related"] == []


# ---------------------------------------------------------------------------
# POST /v1/recall
# ---------------------------------------------------------------------------

class TestRecallRoute:
    def test_recall_returns_notes_text(self, client) -> None:
        resp = client.post("/v1/recall", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "notes" in data
        assert isinstance(data["notes"], str)

    def test_recall_with_query_filter(self, client) -> None:
        resp = client.post("/v1/recall", json={"query": "signal dispatch"})
        assert resp.status_code == 200
        assert "notes" in resp.json()

    def test_recall_with_tags_filter(self, client) -> None:
        resp = client.post("/v1/recall", json={"tags": ["middleware"]})
        assert resp.status_code == 200

    def test_recall_with_priority_filter(self, client) -> None:
        resp = client.post("/v1/recall", json={"priority": "high"})
        assert resp.status_code == 200

    def test_recall_with_limit(self, client) -> None:
        resp = client.post("/v1/recall", json={"limit": 5})
        assert resp.status_code == 200

    def test_recall_processing_ms_present(self, client) -> None:
        resp = client.post("/v1/recall", json={})
        assert "processing_ms" in resp.json()

    # UPG-HOOK-INJECT-OBSERVABILITY: hook_event is the caller-declared field
    # that lets the daemon count hook-driven injections (never inferred from
    # content — see RecallRequest.hook_event).
    @pytest.mark.parametrize("hook_event", ["SessionStart", "UserPromptSubmit", "PreToolUse"])
    def test_recall_valid_hook_event_accepted(self, client, hook_event) -> None:
        resp = client.post("/v1/recall", json={"hook_event": hook_event})
        assert resp.status_code == 200

    def test_recall_invalid_hook_event_rejected(self, client) -> None:
        resp = client.post("/v1/recall", json={"hook_event": "PreCompact"})
        assert resp.status_code == 422

    def test_recall_forwards_hook_event_to_service(self, client) -> None:
        client.post("/v1/recall", json={"boot": True, "hook_event": "SessionStart"})
        from api import app
        # UPG-TEST-SIGNATURE-DRIFT: assert only the kwargs this test is about
        # (hook_event forwarding + boot), validated against the real
        # VectrService.recall signature — so a new kwarg added to the seam does
        # not redden this test, but a renamed/removed param is caught precisely.
        assert_seam_call(
            app.state.service.recall, VectrService.recall,
            boot=True, hook_event="SessionStart",
        )

    def test_recall_without_hook_event_forwards_none(self, client) -> None:
        client.post("/v1/recall", json={})
        from api import app
        assert app.state.service.recall.call_args.kwargs["hook_event"] is None

    # TRIGGER-ENGINE wave 2a: session_id/events (RecallRequest, app/models.py).
    @pytest.mark.parametrize("event", ["session-start", "prompt-submit", "pre-edit", "pre-run", "pre-commit", "post-compaction"])
    def test_recall_valid_event_accepted(self, client, event) -> None:
        resp = client.post("/v1/recall", json={"events": [event]})
        assert resp.status_code == 200

    def test_recall_invalid_event_rejected(self, client) -> None:
        resp = client.post("/v1/recall", json={"events": ["not-a-real-event"]})
        assert resp.status_code == 422

    def test_recall_forwards_session_id_and_events_to_service(self, client) -> None:
        client.post("/v1/recall", json={"session_id": "sess-1", "events": ["prompt-submit"]})
        from api import app
        kwargs = app.state.service.recall.call_args.kwargs
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["events"] == ["prompt-submit"]

    def test_recall_without_session_id_or_events_forwards_none(self, client) -> None:
        client.post("/v1/recall", json={})
        from api import app
        kwargs = app.state.service.recall.call_args.kwargs
        assert kwargs["session_id"] is None
        assert kwargs["events"] is None


# ---------------------------------------------------------------------------
# POST /v1/snapshot
# ---------------------------------------------------------------------------

class TestSnapshotRoute:
    def test_snapshot_returns_snapshot_id(self, client) -> None:
        resp = client.post("/v1/snapshot", json={"label": "phase1-complete"})
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshot_id" in data
        assert data["snapshot_id"] is not None

    def test_snapshot_echoes_label(self, client) -> None:
        resp = client.post("/v1/snapshot", json={"label": "my-label"})
        assert resp.json()["label"] == "my-label"

    def test_snapshot_missing_label_returns_422(self, client) -> None:
        resp = client.post("/v1/snapshot", json={})
        assert resp.status_code == 422

    def test_snapshot_processing_ms_present(self, client) -> None:
        resp = client.post("/v1/snapshot", json={"label": "test"})
        assert "processing_ms" in resp.json()


# ---------------------------------------------------------------------------
# GET /v1/resume (UPG-RESUME-SURFACE)
# ---------------------------------------------------------------------------

class TestResumeRoute:
    def test_resume_returns_200_with_formatted_field(self, client) -> None:
        resp = client.get("/v1/resume")
        assert resp.status_code == 200
        data = resp.json()
        assert "formatted" in data
        assert "last_task" in data
        assert "gotchas" in data
        assert "snapshot" in data
        assert "processing_ms" in data

    def test_resume_empty_workspace_fields_are_empty(self, client) -> None:
        data = client.get("/v1/resume").json()
        assert data["last_task"] is None
        assert data["gotchas"] == []
        assert data["snapshot"] is None

    def test_resume_end_to_end_with_real_store(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "finish the resume surface", "kind": "task", "priority": "high",
        })
        client.post("/v1/remember", json={
            "content": "watch for the flaky snapshot test", "kind": "gotcha",
            "anchors": ["tests/test_flaky.py"],
        })
        client.post("/v1/snapshot", json={"label": "checkpoint-1"})

        resp = client.get("/v1/resume")
        assert resp.status_code == 200
        data = resp.json()

        assert data["last_task"] is not None
        assert data["last_task"]["title"] == "finish the resume surface"
        assert data["last_task"]["kind"] == "task"

        assert len(data["gotchas"]) == 1
        assert data["gotchas"][0]["anchors"] == ["tests/test_flaky.py"]

        assert data["snapshot"]["label"] == "checkpoint-1"
        assert data["snapshot"]["note_count"] == 2

        assert "finish the resume surface" in data["formatted"]
        assert "checkpoint-1" in data["formatted"]
        assert "tests/test_flaky.py" in data["formatted"]

    def test_resume_uses_cli_expand_hint(self, client_real_memory) -> None:
        """The REST route renders with surface='cli' (`vectr recall --id N`)
        — REST is the human/script terminal surface, mirroring how
        POST /v1/recall's CLI caller sets surface='cli' itself (here the
        REST route fixes it, since GET /v1/resume takes no body)."""
        client = client_real_memory
        client.post("/v1/remember", json={"content": "a task", "kind": "task", "priority": "high"})
        formatted = client.get("/v1/resume").json()["formatted"]
        assert "vectr recall --id" in formatted
        assert "vectr_recall(note_id=" not in formatted

    def test_resume_only_gotchas_omits_other_sections(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={"content": "just a gotcha", "kind": "gotcha"})
        data = client.get("/v1/resume").json()
        assert data["last_task"] is None
        assert data["snapshot"] is None
        assert len(data["gotchas"]) == 1


# ---------------------------------------------------------------------------
# POST /v1/trigger/reset (TRIGGER-ENGINE wave 2a)
# ---------------------------------------------------------------------------

class TestTriggerResetRoute:
    def test_reset_returns_reset_true(self, client) -> None:
        resp = client.post("/v1/trigger/reset", json={"session_id": "sess-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reset"] is True
        assert "processing_ms" in data

    def test_reset_forwards_session_id_to_service(self, client) -> None:
        client.post("/v1/trigger/reset", json={"session_id": "sess-1"})
        from api import app
        app.state.service.reset_trigger_ledger.assert_called_with("sess-1")

    def test_reset_with_no_session_id_is_a_no_op_not_an_error(self, client) -> None:
        resp = client.post("/v1/trigger/reset", json={})
        assert resp.status_code == 200
        from api import app
        app.state.service.reset_trigger_ledger.assert_called_with(None)

    def test_reset_never_503s_in_search_only_mode(self, client) -> None:
        """The per-session ledger lives in VectrService itself, not the
        working-memory store — there is nothing to gate on search-only mode."""
        from api import app
        app.state.service.search_only = True
        try:
            resp = client.post("/v1/trigger/reset", json={"session_id": "sess-1"})
            assert resp.status_code == 200
        finally:
            app.state.service.search_only = False


# ---------------------------------------------------------------------------
# GET /v1/evict-hint
# ---------------------------------------------------------------------------

class TestEvictHintRoute:
    def test_evict_hint_returns_hint_and_should_evict(self, client) -> None:
        resp = client.get("/v1/evict-hint")
        assert resp.status_code == 200
        data = resp.json()
        assert "hint" in data
        assert "should_evict" in data

    def test_evict_hint_should_evict_is_bool(self, client) -> None:
        data = client.get("/v1/evict-hint").json()
        assert isinstance(data["should_evict"], bool)

    def test_evict_hint_text_non_empty(self, client) -> None:
        data = client.get("/v1/evict-hint").json()
        assert isinstance(data["hint"], str)
        assert len(data["hint"]) > 0

    def test_evict_hint_route_passes_on_demand(self, client) -> None:
        # UPG-7.2 (B13): the explicit GET is a deliberate ask, so the route must
        # request the on-demand (eviction-focused) framing rather than the
        # auto-footer's remember alarm. MCP green != REST green (R10): assert the
        # route threads on_demand=True to the service. (The framing content
        # itself is unit-tested in TestEvictionHintOnDemandUPG72.)
        svc = client.app.state.service
        svc.eviction_hint.reset_mock()
        client.get("/v1/evict-hint")
        svc.eviction_hint.assert_called_once_with(on_demand=True)


# ---------------------------------------------------------------------------
# Cross-request persistence — THE key integration tests
# ---------------------------------------------------------------------------

class TestMemoryCrossRequest:
    """
    These tests mirror the two-phase POC benchmark flow exactly:
      Request 1 (Phase 1): POST /v1/remember
      Request 2 (Phase 2): POST /v1/recall  → must return what Phase 1 stored

    Uses client_real_memory fixture: real WorkingContextStore, mocked search.
    """

    def test_remember_then_recall_rest_round_trip(self, client_real_memory) -> None:
        client = client_real_memory
        resp = client.post("/v1/remember", json={
            "content": "Field.contribute_to_class at django/db/models/fields/__init__.py:770",
            "tags": ["field"],
            "priority": "high",
        })
        assert resp.status_code == 200

        recall_resp = client.post("/v1/recall", json={})
        notes = recall_resp.json()["notes"]
        assert "contribute_to_class" in notes
        assert "django/db/models/fields" in notes

    def test_remember_with_agent_shows_attribution_on_recall(self, client_real_memory) -> None:
        """UPG-SUBAGENT-MEMORY: a note stored with `agent` renders an attribution
        tag in the recall index line; the orchestrator recalling notes written by
        a different session/subagent sees who authored each one."""
        client = client_real_memory
        resp = client.post("/v1/remember", json={
            "content": "parser bug found in the tokenizer",
            "agent": "coder-2",
        })
        assert resp.status_code == 200

        recall_resp = client.post("/v1/recall", json={})
        notes = recall_resp.json()["notes"]
        assert "(coder-2)" in notes

    def test_remember_without_agent_recall_unchanged(self, client_real_memory) -> None:
        """Absent `agent` renders exactly as before this feature — no stray tag."""
        client = client_real_memory
        resp = client.post("/v1/remember", json={"content": "no attribution here"})
        assert resp.status_code == 200

        recall_resp = client.post("/v1/recall", json={})
        notes = recall_resp.json()["notes"]
        assert "no attribution here" in notes or "no attribution" in notes
        assert "()" not in notes

    def test_multiple_remember_then_recall_all_returned(self, client_real_memory) -> None:
        client = client_real_memory
        findings = [
            "BaseHandler.load_middleware builds the middleware stack",
            "Middleware must set async_capable = True for ASGI",
            "HttpResponseTooManyRequests at django/http/response.py",
        ]
        for content in findings:
            client.post("/v1/remember", json={"content": content})

        notes = client.post("/v1/recall", json={}).json()["notes"]
        for finding in findings:
            first_word = finding.split()[0]
            assert first_word in notes, f"'{first_word}' missing from recalled notes"

    def test_recall_empty_before_any_remember(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/recall", json={})
        assert "No working notes found" in resp.json()["notes"]

    def test_tag_filter_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={"content": "signal dispatch path", "tags": ["signal"]})
        client.post("/v1/remember", json={"content": "middleware chain", "tags": ["middleware"]})

        notes = client.post("/v1/recall", json={"tags": ["signal"]}).json()["notes"]
        assert "signal dispatch" in notes
        assert "middleware chain" not in notes

    def test_priority_filter_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={"content": "high priority finding", "priority": "high"})
        client.post("/v1/remember", json={"content": "low priority note", "priority": "low"})

        notes = client.post("/v1/recall", json={"priority": "high"}).json()["notes"]
        assert "high priority finding" in notes
        assert "low priority note" not in notes

    def test_snapshot_then_recall_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={"content": "key finding for snapshot"})
        snap_resp = client.post("/v1/snapshot", json={"label": "phase1-done"})
        assert snap_resp.status_code == 200
        assert snap_resp.json()["snapshot_id"] is not None

        # Notes still accessible after snapshot
        notes = client.post("/v1/recall", json={}).json()["notes"]
        assert "key finding for snapshot" in notes

    def test_limit_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        for i in range(8):
            client.post("/v1/remember", json={"content": f"finding number {i}"})
        notes_text = client.post("/v1/recall", json={"limit": 3}).json()["notes"]
        # Rough check: with limit=3 the output is shorter than with no limit
        notes_all = client.post("/v1/recall", json={}).json()["notes"]
        assert len(notes_text) <= len(notes_all)

    def test_recall_note_id_increments(self, client_real_memory) -> None:
        client = client_real_memory
        r1 = client.post("/v1/remember", json={"content": "first"}).json()
        r2 = client.post("/v1/remember", json={"content": "second"}).json()
        assert r2["note_id"] > r1["note_id"]

    def test_kind_filter_via_rest(self, client_real_memory) -> None:
        """UPG-9.3: kind round-trips through REST and recall can filter on it."""
        client = client_real_memory
        client.post("/v1/remember", json={"content": "never push to main", "kind": "directive"})
        client.post("/v1/remember", json={"content": "just a plain finding"})

        directives = client.post("/v1/recall", json={"kind": "directive"}).json()["notes"]
        assert "never push to main" in directives
        assert "just a plain finding" not in directives

    def test_invalid_kind_rejected_via_rest(self, client_real_memory) -> None:
        """An unknown kind is a 422 validation error, not a silent finding."""
        resp = client_real_memory.post("/v1/remember", json={"content": "x", "kind": "bogus"})
        assert resp.status_code == 422

    def test_min_similarity_out_of_range_rejected(self, client_real_memory) -> None:
        """UPG-5.1: min_similarity is bounded to [0,1] at the REST boundary."""
        assert client_real_memory.post("/v1/recall", json={"min_similarity": 1.5}).status_code == 422
        assert client_real_memory.post("/v1/recall", json={"min_similarity": -0.1}).status_code == 422
        assert client_real_memory.post("/v1/recall", json={"min_similarity": 0.5}).status_code == 200

    def test_boot_recall_empty_workspace_returns_blank(self, client_real_memory) -> None:
        """UPG-9.2: boot recall on a 0-note workspace returns '' with 200 — never errors."""
        resp = client_real_memory.post("/v1/recall", json={"boot": True})
        assert resp.status_code == 200
        assert resp.json()["notes"] == ""

    def test_boot_recall_returns_directives_and_high_tasks_ignoring_query(self, client_real_memory) -> None:
        """UPG-9.2: boot returns directives + high tasks verbatim regardless of query."""
        client = client_real_memory
        client.post("/v1/remember", json={"content": "never push to main", "kind": "directive"})
        client.post("/v1/remember", json={"content": "sprint goal", "kind": "task", "priority": "high"})
        client.post("/v1/remember", json={"content": "an ordinary finding"})

        notes = client.post("/v1/recall", json={"boot": True, "query": "totally unrelated topic"}).json()["notes"]
        assert "never push to main" in notes
        assert "sprint goal" in notes
        assert "an ordinary finding" not in notes


# ---------------------------------------------------------------------------
# TRIGGER-ENGINE wave 2a — the live engine, end to end through REST
# (client_real_memory: real WorkingContextStore, no daemon/hook involved —
# these tests stand in for the hook-pipeline's REST leg; main.py's cmd_hook
# tests in tests/test_main.py cover the stdin-JSON -> payload leg.)
# ---------------------------------------------------------------------------

class TestTriggerEngineLiveViaRest:
    def test_post_compaction_only_trigger_is_silent_on_plain_boot_but_fires_when_merged(self, client_real_memory) -> None:
        """A note whose ONLY explicit trigger is post-compaction is not part
        of any kind-default bundle — a plain SessionStart (events omitted,
        implying ['session-start']) must not deliver it; merging
        events=['session-start','post-compaction'] (the compact-source
        SessionStart call) must."""
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "post-compaction-only reminder",
            "kind": "finding", "triggers": [{"event": "post-compaction"}],
        })
        plain = client.post("/v1/recall", json={"boot": True}).json()["notes"]
        assert "post-compaction-only reminder" not in plain

        merged = client.post(
            "/v1/recall", json={"boot": True, "events": ["session-start", "post-compaction"]},
        ).json()["notes"]
        assert "post-compaction-only reminder" in merged

    def test_pre_edit_gotcha_with_explicit_trigger_is_not_double_injected(self, client_real_memory) -> None:
        """The engine (explicit pre-edit trigger) and the legacy content-match
        recall_for_path() would BOTH match this note (it mentions auth.py) —
        the note_id dedup must render it exactly once, not twice."""
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "auth.py: verify_token() must check expiry before signature",
            "kind": "gotcha", "anchors": ["src/auth.py"],
            "triggers": [{"path": "src/auth.py", "event": "pre-edit"}],
        })
        notes = client.post("/v1/recall", json={"file_path": "src/auth.py"}).json()["notes"]
        assert notes.count("verify_token() must check expiry") == 1

    def test_explicit_prompt_submit_trigger_fires_via_events_merged_with_query(self, client_real_memory) -> None:
        """No kind's default bundle uses 'prompt-submit' — only an explicit
        override does, and only when the caller passes events=['prompt-submit']
        (main.py's UserPromptSubmit hook)."""
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "always check the retry budget before a network call",
            "kind": "finding", "triggers": [{"event": "prompt-submit"}],
        })
        without_events = client.post(
            "/v1/recall", json={"query": "totally unrelated topic"},
        ).json()["notes"]
        assert "retry budget" not in without_events

        with_events = client.post(
            "/v1/recall", json={"query": "totally unrelated topic", "events": ["prompt-submit"]},
        ).json()["notes"]
        assert "retry budget" in with_events

    def test_events_is_a_no_op_for_a_note_with_no_matching_explicit_trigger(self, client_real_memory) -> None:
        """Passing events=[...] must never change behaviour for a note that
        has no explicit trigger for that event — every caller that predates
        this wave (events always None) is completely unaffected."""
        client = client_real_memory
        client.post("/v1/remember", json={"content": "plain finding with no triggers"})
        notes = client.post(
            "/v1/recall", json={"query": "plain finding", "events": ["prompt-submit"]},
        ).json()["notes"]
        assert "plain finding with no triggers" in notes

    def test_session_scope_note_is_visible_only_to_its_writing_session_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "ephemeral scratch note", "session_id": "sess-a", "scope": "session",
        })
        assert "ephemeral scratch note" in client.post(
            "/v1/recall", json={"session_id": "sess-a"}).json()["notes"]
        assert "ephemeral scratch note" not in client.post(
            "/v1/recall", json={"session_id": "sess-b"}).json()["notes"]
        assert "ephemeral scratch note" not in client.post(
            "/v1/recall", json={}).json()["notes"]

    # memoization-l1-capture-design §5 — command-family trigger lane
    def test_command_trigger_fires_on_matching_normalized_verb(self, client_real_memory) -> None:
        """The 'command' trigger axis globs against the NORMALIZED verb
        (app/cmdnorm.py), not the raw command string — 'pytest -q tests/'
        must match a trigger declared against the 'pytest' verb."""
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "pytest must use ./.venv/bin/python, not global python",
            "kind": "operational", "triggers": [{"command": "pytest"}],
        })
        notes = client.post(
            "/v1/recall", json={"command": "pytest -q tests/test_foo.py"}).json()["notes"]
        assert "must use ./.venv/bin/python" in notes

    def test_command_trigger_is_silent_for_a_different_verb(self, client_real_memory) -> None:
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "pytest must use ./.venv/bin/python, not global python",
            "kind": "operational", "triggers": [{"command": "pytest"}],
        })
        notes = client.post(
            "/v1/recall", json={"command": "git status"}).json()["notes"]
        assert "must use ./.venv/bin/python" not in notes

    def test_command_trigger_does_not_fire_via_a_query_that_names_the_command(self, client_real_memory) -> None:
        """The 'command' axis only ever matches PreToolUse's normalized
        VERB (app/cmdnorm.py), passed through recall()'s dedicated `command`
        field — never a query string. Mentioning 'pytest' in a semantic
        query must not fire the trigger through the query/events path (the
        no-query-heuristics rule: only the caller-declared `command` field,
        never parsed prompt content, drives this axis)."""
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "pytest must use ./.venv/bin/python, not global python",
            "kind": "operational", "triggers": [{"command": "pytest"}],
        })
        notes = client.post(
            "/v1/recall", json={"query": "how do I run pytest", "events": ["prompt-submit"]},
        ).json()["notes"]
        assert "must use ./.venv/bin/python" not in notes

    def test_two_surfaces_same_turn_inject_once_via_rest(self, client_real_memory) -> None:
        """N4 (memoization-l1-capture-design §5.3/§5.4, serving-policy
        review): the turn-scoped injection ledger must be threaded through
        the REST /v1/recall path, not only exercised at the store level —
        a note with two live trigger axes (session-start and pre-edit) that
        both match within the same turn (same session_id, no reset between
        calls, exactly mirroring an editor that fires SessionStart then a
        same-turn PreToolUse before the next UserPromptSubmit) must be
        delivered by the FIRST surface only; the second surface's own,
        different axis match is turn-deduped, not re-delivered."""
        client = client_real_memory
        client.post("/v1/remember", json={
            "content": "auth.py: verify_token() must check expiry before signature",
            "kind": "finding",
            "triggers": [{"event": "session-start"}, {"path": "src/auth.py", "event": "pre-edit"}],
        })
        boot_notes = client.post(
            "/v1/recall", json={"boot": True, "session_id": "sess-n4"},
        ).json()["notes"]
        assert "verify_token() must check expiry" in boot_notes

        edit_notes = client.post(
            "/v1/recall", json={"file_path": "src/auth.py", "session_id": "sess-n4"},
        ).json()["notes"]
        assert "verify_token() must check expiry" not in edit_notes


# ---------------------------------------------------------------------------
# POST /v1/forget
# ---------------------------------------------------------------------------

class TestForgetRoute:
    def test_forget_note_id_deletes_only_that_note(self, client_real_memory) -> None:
        r1 = client_real_memory.post("/v1/remember", json={"content": "note to delete"})
        r2 = client_real_memory.post("/v1/remember", json={"content": "note to keep"})
        nid1, nid2 = r1.json()["note_id"], r2.json()["note_id"]

        resp = client_real_memory.post("/v1/forget", json={"note_id": nid1})
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1
        assert resp.json()["found"] is True

        remaining = client_real_memory.post(
            "/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "note to keep" in remaining
        assert "note to delete" not in remaining

    def test_forget_note_id_not_found(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/forget", json={"note_id": 99999})
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0
        assert resp.json()["found"] is False

    def test_forget_all_true_clears_workspace(self, client_real_memory) -> None:
        client_real_memory.post("/v1/remember", json={"content": "a"})
        client_real_memory.post("/v1/remember", json={"content": "b"})
        resp = client_real_memory.post("/v1/forget", json={"all": True})
        assert resp.status_code == 200
        assert resp.json()["deleted"] >= 2

    def test_forget_no_arguments_is_422_and_deletes_nothing(self, client_real_memory) -> None:
        # Data-loss regression guard (2026-07-02): bare forget must never wipe the store.
        r = client_real_memory.post("/v1/remember", json={"content": "survivor note"})
        assert r.status_code == 200
        resp = client_real_memory.post("/v1/forget", json={})
        assert resp.status_code == 422
        remaining = client_real_memory.post(
            "/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "survivor note" in remaining


# ---------------------------------------------------------------------------
# POST /v1/promote (TRIGGER-ENGINE wave 1, bm2-design-skeleton.md §5)
# ---------------------------------------------------------------------------

class TestPromoteRoute:
    def test_promote_returns_200_with_mocked_service(self, client) -> None:
        resp = client.post("/v1/promote", json={"note_id": 1, "to": "agent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["note_id"] == 1
        assert data["provenance"] == "agent"
        assert "processing_ms" in data

    def test_promote_invalid_to_value_returns_422(self, client) -> None:
        resp = client.post("/v1/promote", json={"note_id": 1, "to": "not-a-real-provenance"})
        assert resp.status_code == 422

    def test_promote_auto_to_agent_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        note_id = client.post(
            "/v1/remember", json={"content": "auto note", "provenance": "auto"}
        ).json()["note_id"]
        resp = client.post("/v1/promote", json={"note_id": note_id, "to": "agent"})
        assert resp.status_code == 200
        assert resp.json()["provenance"] == "agent"

    def test_promote_skip_a_rank_returns_422(self, client_real_memory) -> None:
        client = client_real_memory
        note_id = client.post(
            "/v1/remember", json={"content": "auto note", "provenance": "auto"}
        ).json()["note_id"]
        resp = client.post("/v1/promote", json={"note_id": note_id, "to": "human"})
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "invalid_promotion"

    def test_promote_nonexistent_note_returns_404(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/promote", json={"note_id": 999999, "to": "agent"})
        assert resp.status_code == 404

    def test_promote_agent_to_human_via_rest(self, client_real_memory) -> None:
        """REST is the user-side promotion surface (a CLI/UI a person operates),
        so unlike the MCP tool it supports the full one-step chain, including
        the final agent -> human step (bm2-design-skeleton.md §5)."""
        client = client_real_memory
        note_id = client.post(
            "/v1/remember", json={"content": "auto note", "provenance": "auto"}
        ).json()["note_id"]
        client.post("/v1/promote", json={"note_id": note_id, "to": "agent"})
        resp = client.post("/v1/promote", json={"note_id": note_id, "to": "human"})
        assert resp.status_code == 200
        assert resp.json()["provenance"] == "human"


# ---------------------------------------------------------------------------
# POST /v1/revoke, POST /v1/reinstate (UPG-MEMORY-STATE-MACHINE §4.2)
# ---------------------------------------------------------------------------

class TestRevokeRoute:
    def test_revoke_returns_200_with_mocked_service(self, client) -> None:
        resp = client.post("/v1/revoke", json={"note_id": 1, "reason": "turned out false"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["note_id"] == 1
        assert "processing_ms" in data

    def test_revoke_missing_reason_returns_422(self, client) -> None:
        resp = client.post("/v1/revoke", json={"note_id": 1})
        assert resp.status_code == 422

    def test_revoke_invalid_actor_returns_422(self, client) -> None:
        resp = client.post(
            "/v1/revoke", json={"note_id": 1, "reason": "wrong", "actor": "system"}
        )
        assert resp.status_code == 422

    def test_revoke_via_rest_with_real_store(self, client_real_memory) -> None:
        client = client_real_memory
        note_id = client.post("/v1/remember", json={"content": "a finding"}).json()["note_id"]
        resp = client.post("/v1/revoke", json={"note_id": note_id, "reason": "wrong", "actor": "human"})
        assert resp.status_code == 200
        recalled = client.post("/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "Previously believed" in recalled

    def test_revoke_nonexistent_note_returns_404(self, client_real_memory) -> None:
        resp = client_real_memory.post(
            "/v1/revoke", json={"note_id": 999999, "reason": "wrong"}
        )
        assert resp.status_code == 404


class TestReinstateRoute:
    def test_reinstate_returns_200_with_mocked_service(self, client) -> None:
        resp = client.post("/v1/reinstate", json={"note_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["note_id"] == 1
        assert "processing_ms" in data

    def test_reinstate_invalid_actor_returns_422(self, client) -> None:
        resp = client.post("/v1/reinstate", json={"note_id": 1, "actor": "system"})
        assert resp.status_code == 422

    def test_reinstate_via_rest_reverses_revoke(self, client_real_memory) -> None:
        client = client_real_memory
        note_id = client.post("/v1/remember", json={"content": "a finding"}).json()["note_id"]
        client.post("/v1/revoke", json={"note_id": note_id, "reason": "wrong"})
        resp = client.post("/v1/reinstate", json={"note_id": note_id, "reason": "verified correct"})
        assert resp.status_code == 200
        recalled = client.post("/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "a finding" in recalled
        assert "Previously believed" not in recalled

    def test_reinstate_nonexistent_note_returns_404(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/reinstate", json={"note_id": 999999})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/supersede (UPG-NOTE-RETIRE-POST-HOC)
# ---------------------------------------------------------------------------

class TestSupersedeRoute:
    def test_supersede_returns_200_with_mocked_service(self, client) -> None:
        resp = client.post(
            "/v1/supersede", json={"note_id": 1, "superseded_by": 2, "reason": "replaced in substance"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["note_id"] == 1
        assert "processing_ms" in data

    def test_supersede_invalid_actor_returns_422(self, client) -> None:
        # 'system' is reserved for the deterministic transitions — deciding a
        # stored note is already-superseded is always a judgment call.
        resp = client.post("/v1/supersede", json={"note_id": 1, "actor": "system"})
        assert resp.status_code == 422

    def test_supersede_missing_successor_returns_422(self, client_real_memory) -> None:
        # Validated BEFORE any mutation: the target must be untouched.
        client = client_real_memory
        note_id = client.post("/v1/remember", json={"content": "a finding"}).json()["note_id"]
        resp = client.post("/v1/supersede", json={"note_id": note_id, "superseded_by": 999999})
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "invalid_supersedence"
        recalled = client.post("/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "a finding" in recalled  # still active — nothing half-applied

    def test_supersede_via_rest_excludes_from_default_recall(self, client_real_memory) -> None:
        client = client_real_memory
        old_id = client.post("/v1/remember", json={"content": "the API returns a list"}).json()["note_id"]
        new_id = client.post("/v1/remember", json={"content": "the API returns a dict"}).json()["note_id"]
        resp = client.post(
            "/v1/supersede",
            json={"note_id": old_id, "superseded_by": new_id, "actor": "human", "reason": "consolidated"},
        )
        assert resp.status_code == 200
        default_recall = client.post("/v1/recall", json={}).json()["notes"]
        assert "the API returns a list" not in default_recall

    def test_supersede_human_actor_accepted_on_agent_provenance_note(self, client_real_memory) -> None:
        # REST is the person-operated surface: actor="human" on an
        # agent-provenance note is legal (only the reverse — agent actor on
        # a human-provenance note — crosses the write boundary).
        client = client_real_memory
        note_id = client.post("/v1/remember", json={"content": "a finding"}).json()["note_id"]
        resp = client.post("/v1/supersede", json={"note_id": note_id, "actor": "human"})
        assert resp.status_code == 200

    def test_reinstate_reverses_supersede_via_rest(self, client_real_memory) -> None:
        client = client_real_memory
        note_id = client.post("/v1/remember", json={"content": "wrongly retired"}).json()["note_id"]
        other_id = client.post("/v1/remember", json={"content": "unrelated"}).json()["note_id"]
        client.post("/v1/supersede", json={"note_id": note_id, "superseded_by": other_id})
        resp = client.post("/v1/reinstate", json={"note_id": note_id, "reason": "retirement was wrong"})
        assert resp.status_code == 200
        recalled = client.post("/v1/recall", json={"detail": "full"}).json()["notes"]
        assert "wrongly retired" in recalled

    def test_supersede_nonexistent_note_returns_404(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/supersede", json={"note_id": 999999})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/pin (UPG-RECALL-MISS-FLOOR part (b)) — explicit Tier 0 membership
# ---------------------------------------------------------------------------

class TestPinRoute:
    def test_pin_returns_200_with_mocked_service(self, client) -> None:
        resp = client.post("/v1/pin", json={"note_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["note_id"] == 1
        assert data["pinned"] is True
        assert "processing_ms" in data

    def test_pin_defaults_pinned_true(self, client) -> None:
        resp = client.post("/v1/pin", json={"note_id": 1})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

    def test_unpin_via_rest(self, client) -> None:
        resp = client.post("/v1/pin", json={"note_id": 1, "pinned": False})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is False

    def test_pin_via_rest_with_real_store(self, client_real_memory) -> None:
        client = client_real_memory
        note_id = client.post("/v1/remember", json={"content": "a finding to pin"}).json()["note_id"]
        resp = client.post("/v1/pin", json={"note_id": note_id})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

    def test_unpin_via_rest_with_real_store(self, client_real_memory) -> None:
        client = client_real_memory
        note_id = client.post("/v1/remember", json={"content": "a finding to pin"}).json()["note_id"]
        client.post("/v1/pin", json={"note_id": note_id})
        resp = client.post("/v1/pin", json={"note_id": note_id, "pinned": False})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is False

    def test_pin_at_write_time_via_remember(self, client_real_memory) -> None:
        """pin=True on /v1/remember is equivalent to a separate /v1/pin call
        right after the write (UPG-RECALL-MISS-FLOOR part (b))."""
        client = client_real_memory
        resp = client.post("/v1/remember", json={"content": "pinned at write time", "pin": True})
        assert resp.status_code == 200
        # A second, independent /v1/pin(False) call must actually flip it back
        # -- proof the write-time pin landed on the real note, not a no-op.
        note_id = resp.json()["note_id"]
        unpin = client.post("/v1/pin", json={"note_id": note_id, "pinned": False})
        assert unpin.status_code == 200
        assert unpin.json()["pinned"] is False

    def test_pin_nonexistent_note_returns_404(self, client_real_memory) -> None:
        resp = client_real_memory.post("/v1/pin", json={"note_id": 999999})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# UPG-SUPERSEDES-GUARD-E2E — a non-human-provenance write may never supersedes-
# tombstone a human-provenance note, end-to-end through the real REST write
# surfaces. No write surface mints provenance='human' directly (by design), so
# the /v1/promote chain (auto -> agent -> human) is the only way to a genuine
# human note; these tests drive that chain, then attack the resulting note with
# a non-human `supersedes` write and assert the store guard fires end-to-end.
# ---------------------------------------------------------------------------

class TestSupersedesHumanProvenanceGuardE2E:
    def _make_human_note(self, client) -> int:
        note_id = client.post(
            "/v1/remember",
            json={"content": "human-reviewed directive: never disable auth", "provenance": "auto"},
        ).json()["note_id"]
        client.post("/v1/promote", json={"note_id": note_id, "to": "agent"})
        resp = client.post("/v1/promote", json={"note_id": note_id, "to": "human"})
        assert resp.status_code == 200
        assert resp.json()["provenance"] == "human"
        return note_id

    def test_agent_write_cannot_supersede_human_note(self, client_real_memory) -> None:
        client = client_real_memory
        human_id = self._make_human_note(client)
        # Default REST provenance is 'agent' — a non-human write.
        resp = client.post(
            "/v1/remember",
            json={"content": "agent tries to override the directive", "supersedes": human_id},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error"] == "invalid_memory_object"
        assert "human" in detail["detail"].lower()

    def test_auto_write_cannot_supersede_human_note(self, client_real_memory) -> None:
        client = client_real_memory
        human_id = self._make_human_note(client)
        resp = client.post(
            "/v1/remember",
            json={"content": "auto capture", "provenance": "auto", "supersedes": human_id},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "invalid_memory_object"

    def test_rejected_supersede_leaves_human_note_live(self, client_real_memory) -> None:
        """The guard raises before any write, so the human note is not
        tombstoned — it still recalls after the rejected supersede attempt."""
        client = client_real_memory
        human_id = self._make_human_note(client)
        client.post(
            "/v1/remember",
            json={"content": "agent tries to override the directive", "supersedes": human_id},
        )
        recalled = client.post(
            "/v1/recall", json={"query": "human-reviewed directive"}
        ).json()["notes"]
        assert "never disable auth" in recalled

    def test_agent_write_may_supersede_non_human_note(self, client_real_memory) -> None:
        """Control: the guard is specific to HUMAN targets. An agent write may
        still supersede an agent-provenance note end-to-end (200), proving the
        rejection above is not a blanket ban on `supersedes`."""
        client = client_real_memory
        agent_id = client.post(
            "/v1/remember", json={"content": "agent note to replace", "provenance": "agent"}
        ).json()["note_id"]
        resp = client.post(
            "/v1/remember",
            json={"content": "newer agent note", "supersedes": agent_id},
        )
        assert resp.status_code == 200
