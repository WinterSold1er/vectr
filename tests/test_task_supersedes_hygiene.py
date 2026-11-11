"""
UPG-TASK-SUPERSEDES-HYGIENE — a memory-hygiene NUDGE, never a lifecycle
change: kind="task" notes never decay, auto-supersede, or auto-expire.

UPG-STATUS-AGE-ONLY-FORGET-NUDGE — that nudge is NEUTRAL inventory framing,
never a WARNING, and never nominates vectr_forget: age alone is not a
deterministic staleness signal, so the only remediation it names is
supersession (a caller judgment about work state).

Coverage:
  - WorkingContextStore.stale_task_summary(): state-based count + oldest id
    (kind + age + tombstone status only, never content).
  - VectrService.stale_task_summary() / status(): wraps the store, exposed
    unconditionally, (0, None) in search-only mode.
  - vectr_status MCP output: neutral line appended above the configured
    threshold, absent below it and when notes are superseded; names only
    supersedes as remediation.
  - /v1/status REST: same fields surfaced through StatusResponse.
  - Config-driven: mutating the threshold changes behaviour without any
    code change.
"""
from __future__ import annotations

import time

import pytest


def _store(tmp_path):
    from agent.working_context_store import WorkingContextStore
    return WorkingContextStore(str(tmp_path))


def _backdate(store, note_id, age_days):
    cutoff = time.time() - age_days * 86400
    with store._conn() as conn:
        conn.execute(
            "UPDATE notes SET created_at = ? WHERE note_id = ?", (cutoff, note_id)
        )


# ---------------------------------------------------------------------------
# Store-level: WorkingContextStore.stale_task_summary()
# ---------------------------------------------------------------------------

class TestStaleTaskSummaryStore:
    def test_no_task_notes_returns_zero_and_none(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (0, None)

    def test_recent_task_note_not_counted(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        store.remember(ws, "current work", kind="task")
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (0, None)

    def test_old_task_note_counted(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "stale checkpoint", kind="task")
        _backdate(store, note_id, age_days=10)
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (1, note_id)

    def test_oldest_id_is_earliest_created_at(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        older_id = store.remember(ws, "oldest checkpoint", kind="task")
        _backdate(store, older_id, age_days=20)
        newer_id = store.remember(ws, "newer stale checkpoint", kind="task")
        _backdate(store, newer_id, age_days=10)
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert count == 2
        assert oldest_id == older_id

    def test_superseded_old_task_note_not_counted(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        old_id = store.remember(ws, "old checkpoint", kind="task")
        _backdate(store, old_id, age_days=10)
        # New task note explicitly supersedes the old one — old is tombstoned
        # (valid_until set) and must stop counting toward staleness.
        store.remember(ws, "current checkpoint", kind="task", supersedes=old_id)
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (0, None)

    def test_forgotten_old_task_note_not_counted(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        old_id = store.remember(ws, "old checkpoint", kind="task")
        _backdate(store, old_id, age_days=10)
        store.forget(ws, old_id)
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (0, None)

    def test_non_task_kind_notes_not_counted(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        gotcha_id = store.remember(ws, "old gotcha", kind="gotcha")
        finding_id = store.remember(ws, "old finding", kind="finding")
        _backdate(store, gotcha_id, age_days=30)
        _backdate(store, finding_id, age_days=30)
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (0, None)

    def test_workspace_isolation(self, tmp_path) -> None:
        store = _store(tmp_path)
        other_ws = str(tmp_path) + "-other"
        note_id = store.remember(other_ws, "stale in other workspace", kind="task")
        _backdate(store, note_id, age_days=30)
        count, oldest_id = store.stale_task_summary(str(tmp_path), min_age_days=7)
        assert (count, oldest_id) == (0, None)

    # UPG-STALE-TASK-SUMMARY-COUNTS-DEAD-STATES: the original predicate
    # was `valid_until IS NULL`, which catches the SUPERSEDE tombstone
    # (an explicit column write) but not the EVENT-fold terminal states
    # REVOKED and EXPIRED (both of which leave `valid_until` NULL by
    # their own contracts — see `revoke_note()` and `purge_expired_notes
    # ()` in agent/working_context_store/_store.py). The new predicate
    # also requires the event-fold state to be "active". These tests
    # pin the new behaviour at the store level: a stale-task hygiene
    # nudge that recommends "consider supersedes=<old id>" for a note
    # the system has already retired (EXPIRED) or the user already
    # judged wrong (REVOKED) is wrong-footed noise that mis-tunes
    # UPG-HYGIENE-THRESHOLD-RETUNE.

    def test_revoked_old_task_note_not_counted(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "wrong checkpoint", kind="task")
        _backdate(store, note_id, age_days=10)
        # REVOKED: appends the `revoked` event; `revoke_note()`'s contract
        # deliberately leaves `valid_until` NULL (anti-memory deterrent
        # rendering must stay visible, so tombstoning would be wrong).
        # The pre-fix predicate missed this — the note was un-superseded
        # in the column sense, so the old count included it.
        store.revoke_note(ws, note_id, reason="judged wrong")
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (0, None)

    def test_expired_old_task_note_not_counted(self, tmp_path) -> None:
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "aged-out checkpoint", kind="task")
        _backdate(store, note_id, age_days=30)
        # EXPIRED: `purge_expired_notes()` appends the `expired` event
        # and deliberately leaves the columns NULL
        # (UPG-MEMORY-DECAY-KIND-SCOPED's append-only invariant). The
        # ttl_days=1 override collapses every kind's effective TTL to
        # 1 day so the 30-day-old backdated note crosses it; the kind
        # baseline (21d for task) would also fire on a 30-day-old note
        # on its own, but the override makes the test independent of
        # the shipped config.
        expired = store.purge_expired_notes(ws, ttl_days=1)
        assert expired == 1
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (0, None)

    def test_reinstated_revoked_old_task_note_counts_again(self, tmp_path) -> None:
        """A revoked note that the user reverses should resume counting
        toward the hygiene nudge — the same way a brand-new active note
        would. This pins the "fold is the authority" contract: reinstated
        brings the folded state back to `active`, and the new predicate
        honours that transition."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "checkpoint I will flip on", kind="task")
        _backdate(store, note_id, age_days=10)
        store.revoke_note(ws, note_id, reason="first judgment")
        # Sanity: revoked is dead to the inventory.
        assert store.stale_task_summary(ws, min_age_days=7) == (0, None)
        # Reinstate brings it back to active.
        store.reinstate_note(ws, note_id, reason="second look, kept it")
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (1, note_id)

    def test_reinstated_expired_old_task_note_counts_again(self, tmp_path) -> None:
        """The same reinstate-after-expiry symmetry: an expired note the
        user later revives via `reinstate_note()` (the SAME `reinstated`
        event reverses every terminal state per
        UPG-MEMORY-STATE-MACHINE §4.2) must count toward the nudge
        again."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        note_id = store.remember(ws, "checkpoint that aged out", kind="task")
        _backdate(store, note_id, age_days=30)
        store.purge_expired_notes(ws, ttl_days=1)
        assert store.stale_task_summary(ws, min_age_days=7) == (0, None)
        store.reinstate_note(ws, note_id, reason="un-expired, still relevant")
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (1, note_id)

    def test_oldest_id_skips_dead_state_note_older_than_live_one(
        self, tmp_path
    ) -> None:
        """The count and the oldest-id subquery are TWO separate
        predicates in the same statement (per the brief), so a fix that
        tightens one and not the other is a regression waiting to
        happen. This test pins BOTH: a dead-state (revoked) note that
        is OLDER than a live active note must not pull the oldest-id
        away from the live one, AND the count must reflect only the
        live set. The two together are the whole acceptance gate — if
        a future change restores the single-column predicate or splits
        the count and oldest-id into two different predicates, this
        test fails immediately."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        # Live note: created 10 days ago, still active.
        live_id = store.remember(ws, "current checkpoint", kind="task")
        _backdate(store, live_id, age_days=10)
        # Dead note: created 20 days ago, REVOKED. Older than the live
        # one but in a terminal state — must NOT count, and must NOT
        # be picked as the oldest-id.
        dead_id = store.remember(ws, "old wrong checkpoint", kind="task")
        _backdate(store, dead_id, age_days=20)
        store.revoke_note(ws, dead_id, reason="wrong")

        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        # The count is 1, the live note's id. The dead note is
        # filtered out on both axes — count AND oldest-id.
        assert count == 1, (
            f"expected 1 live stale task note, got {count}; "
            f"the revoked note was {dead_id} (backdated 20d), "
            f"the live one is {live_id} (backdated 10d)"
        )
        assert oldest_id == live_id, (
            f"oldest_id must skip the dead-state note; "
            f"got {oldest_id}, expected {live_id}"
        )

    def test_oldest_id_skips_expired_note_older_than_live_one(
        self, tmp_path
    ) -> None:
        """Same shape as the revoked test above, but for the EXPIRED
        terminal state — pinning the same two-axis invariant for a
        different dead state, since the brief's "the count and the
        oldest-id subquery are two separate predicates" warning
        applies equally to both."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        live_id = store.remember(ws, "current checkpoint", kind="task")
        _backdate(store, live_id, age_days=10)
        expired_id = store.remember(ws, "old aged-out checkpoint", kind="task")
        _backdate(store, expired_id, age_days=20)
        # TTL must fall BETWEEN the two ages. `purge_expired_notes` expires
        # every note past the cutoff, so a ttl below 10 days would expire the
        # "live" note too, leaving nothing to compare against — the assertion
        # would then be measuring an empty set rather than the skip behaviour.
        newly_expired = store.purge_expired_notes(ws, ttl_days=15)
        assert newly_expired == 1, (
            f"setup expired {newly_expired} notes; exactly the 20d note must "
            f"expire and the 10d one must stay active for this test to mean "
            f"anything"
        )
        assert store._note_event_states_by_ids(ws, [live_id])[live_id][
            "state"
        ] == "active"

        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert count == 1, (
            f"expected 1 live stale task note, got {count}; "
            f"the expired note was {expired_id} (backdated 20d), "
            f"the live one is {live_id} (backdated 10d)"
        )
        assert oldest_id == live_id, (
            f"oldest_id must skip the expired note; "
            f"got {oldest_id}, expected {live_id}"
        )

    def test_active_count_unchanged_for_pre_migration_notes(self, tmp_path) -> None:
        """Notes with NO `note_events` rows at all (pre-migration data)
        fold to `active` by `fold()`'s own default — the same state a
        brand-new `created` event would produce. The new predicate
        must NOT regress this: a backdated task note with no event
        log must still count toward the hygiene nudge exactly the way
        it did before the change. Backstop for the "we accidentally
        excluded old notes" failure mode."""
        store = _store(tmp_path)
        ws = str(tmp_path)
        # Insert directly: skip `remember()`'s `created` event so this
        # note has an EMPTY `note_events` log, simulating a row from
        # before the event-log feature shipped. Every NOT-NULL column
        # with no default gets an explicit value; the rest (tags,
        # decay_score, author_id, author_trust_score, valid_from,
        # code_hash, pinned) take the schema's own defaults.
        with store._conn() as conn:
            conn.execute(
                "INSERT INTO notes (workspace, content, priority, kind,"
                " created_at, last_accessed) VALUES (?, '', 'medium',"
                " 'task', ?, ?)",
                (ws, time.time() - 30 * 86400, time.time()),
            )
            pre_migration_id = conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]
        count, oldest_id = store.stale_task_summary(ws, min_age_days=7)
        assert (count, oldest_id) == (1, pre_migration_id)


# ---------------------------------------------------------------------------
# Service-level: VectrService.stale_task_summary() / status()
# ---------------------------------------------------------------------------

class TestServiceStaleTaskSummary:
    def _make_service(self, tmp_path, monkeypatch, **kwargs):
        from unittest.mock import patch
        from agent import indexer as idx_module
        from tests.conftest import _DummyEmbedProvider

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())

        with patch("integrations.vscode_bridge.configure_all"), \
             patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
             patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
            from app.service import VectrService
            svc = VectrService(workspace_root=str(tmp_path), **kwargs)
        return svc

    def test_zero_when_no_stale_tasks(self, tmp_path, monkeypatch) -> None:
        svc = self._make_service(tmp_path, monkeypatch)
        assert svc.stale_task_summary() == (0, None)

    def test_reflects_backdated_task_notes(self, tmp_path, monkeypatch) -> None:
        svc = self._make_service(tmp_path, monkeypatch)
        note_id = svc._context_store.remember(
            svc._workspace_root, "stale checkpoint", kind="task"
        )
        _backdate(svc._context_store, note_id, age_days=10)
        assert svc.stale_task_summary() == (1, note_id)

    def test_status_dict_includes_stale_task_fields(self, tmp_path, monkeypatch) -> None:
        svc = self._make_service(tmp_path, monkeypatch)
        status = svc.status()
        assert "stale_task_count" in status
        assert "stale_task_oldest_id" in status
        assert status["stale_task_count"] == 0
        assert status["stale_task_oldest_id"] is None

    def test_search_only_mode_returns_zero_and_none(self, tmp_path, monkeypatch) -> None:
        svc = self._make_service(tmp_path, monkeypatch, search_only=True)
        assert svc.stale_task_summary() == (0, None)

    def test_boot_recall_appends_stale_task_nudge(self, tmp_path, monkeypatch) -> None:
        # UPG-NUDGE-HOOK-PATH-UNREACHABLE (B9): the SessionStart injection path
        # (recall boot=True) must carry the stale-task nudge, since the shipped
        # guidance steers agents away from vectr_status where it otherwise lives.
        from agent.config import MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        svc = self._make_service(tmp_path, monkeypatch)
        oldest = None
        for i in range(MEMORY_HYGIENE_STALE_TASK_WARN_COUNT):
            nid = svc._context_store.remember(
                svc._workspace_root, f"stale checkpoint {i}", kind="task"
            )
            _backdate(svc._context_store, nid, age_days=10)
            if oldest is None:
                oldest = nid
        count, oldest_id = svc.stale_task_summary()
        assert count >= MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        boot_text = svc.recall("", boot=True, session_id="sess-boot")
        assert "Memory hygiene" in boot_text
        assert f"#{oldest_id}" in boot_text

    def test_boot_recall_nudge_never_nominates_forget(self, tmp_path, monkeypatch) -> None:
        # UPG-STATUS-AGE-ONLY-FORGET-NUDGE: the boot-injection nudge must offer
        # supersession as the remediation and never nominate vectr_forget on an
        # age threshold.
        from agent.config import MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        svc = self._make_service(tmp_path, monkeypatch)
        for i in range(MEMORY_HYGIENE_STALE_TASK_WARN_COUNT):
            nid = svc._context_store.remember(
                svc._workspace_root, f"stale checkpoint {i}", kind="task"
            )
            _backdate(svc._context_store, nid, age_days=10)
        boot_text = svc.recall("", boot=True, session_id="sess-forget-check")
        assert "vectr_forget" not in boot_text
        assert "supersedes" in boot_text

    def test_boot_recall_no_nudge_below_threshold(self, tmp_path, monkeypatch) -> None:
        svc = self._make_service(tmp_path, monkeypatch)
        nid = svc._context_store.remember(svc._workspace_root, "one stale task", kind="task")
        _backdate(svc._context_store, nid, age_days=10)
        # A single stale task is below the warn count → no nudge.
        boot_text = svc.recall("", boot=True, session_id="sess-boot2")
        assert "Memory hygiene" not in boot_text


# ---------------------------------------------------------------------------
# MCP-level: vectr_status stale-task line (mocked svc.status(), UPG-9x pattern)
# ---------------------------------------------------------------------------

class TestMCPStatusStaleTaskWarning:
    def _base_status(self, **overrides):
        status = {
            "indexed_files": 10, "total_chunks": 50,
            "last_indexed": "2026-01-01T00:00:00Z",
            "embed_model": "test-model", "workspace_root": "/repo",
            "symbol_count": 0, "notes_count": 0, "languages": [],
            "grammars_unavailable": [],
        }
        status.update(overrides)
        return status

    def _call(self, status_dict):
        from unittest.mock import MagicMock
        from integrations.mcp_server._dispatch import handle_tools_call

        svc = MagicMock()
        svc.status.return_value = status_dict
        svc.count_notes.return_value = status_dict.get("notes_count", 0)
        svc.suggest_instruction_style.return_value = "additive"
        svc._eviction_advisor = MagicMock()

        result = handle_tools_call("vectr_status", {}, svc, session_id="test")
        return result["content"][0]["text"]

    # UPG-STATUS-AGE-ONLY-FORGET-NUDGE: the line is neutral inventory
    # framing, not a WARNING — age alone is never a staleness verdict, and
    # the one destructive verb is never nominated on it.

    def test_line_present_above_threshold(self) -> None:
        from agent.config import MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        text = self._call(self._base_status(
            stale_task_count=MEMORY_HYGIENE_STALE_TASK_WARN_COUNT,
            stale_task_oldest_id=42,
        ))
        assert "#42" in text
        assert "checkpoint(s)" in text
        assert "still active" in text

    def test_line_is_not_a_warning(self) -> None:
        """The task-note line must not carry WARNING severity. With this mock
        status dict no other WARNING line can fire (grammars_unavailable is
        empty and no failure keys are set), so any WARNING in the output
        would have to be this line's."""
        from agent.config import MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        text = self._call(self._base_status(
            stale_task_count=MEMORY_HYGIENE_STALE_TASK_WARN_COUNT,
            stale_task_oldest_id=42,
        ))
        assert "WARNING" not in text

    def test_line_never_nominates_forget(self) -> None:
        from agent.config import MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        text = self._call(self._base_status(
            stale_task_count=MEMORY_HYGIENE_STALE_TASK_WARN_COUNT + 100,
            stale_task_oldest_id=42,
        ))
        assert "vectr_forget" not in text

    def test_line_names_supersede_as_the_remediation(self) -> None:
        from agent.config import MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        text = self._call(self._base_status(
            stale_task_count=MEMORY_HYGIENE_STALE_TASK_WARN_COUNT,
            stale_task_oldest_id=7,
        ))
        assert 'supersedes=<old id>' in text
        # The successor checkpoint guidance steers toward resume visibility too.
        assert 'priority="high"' in text

    def test_line_absent_below_threshold(self) -> None:
        from agent.config import MEMORY_HYGIENE_STALE_TASK_WARN_COUNT
        text = self._call(self._base_status(
            stale_task_count=MEMORY_HYGIENE_STALE_TASK_WARN_COUNT - 1,
            stale_task_oldest_id=42,
        ))
        assert "checkpoint(s)" not in text

    def test_line_absent_when_count_zero(self) -> None:
        text = self._call(self._base_status(stale_task_count=0, stale_task_oldest_id=None))
        assert "checkpoint(s)" not in text

    def test_line_absent_when_key_missing_entirely(self) -> None:
        """Backward-compat: an older mock/status dict without the new keys
        must not error and must not warn (defaults to 0)."""
        text = self._call(self._base_status())
        assert "checkpoint(s)" not in text


# ---------------------------------------------------------------------------
# Config-driven: mutating the threshold changes behaviour, not the code
# ---------------------------------------------------------------------------

class TestConfigDrivenThreshold:
    def _call(self, status_dict):
        from unittest.mock import MagicMock
        from integrations.mcp_server._dispatch import handle_tools_call

        svc = MagicMock()
        svc.status.return_value = status_dict
        svc.count_notes.return_value = status_dict.get("notes_count", 0)
        svc.suggest_instruction_style.return_value = "additive"
        svc._eviction_advisor = MagicMock()

        result = handle_tools_call("vectr_status", {}, svc, session_id="test")
        return result["content"][0]["text"]

    def test_lowering_threshold_makes_same_count_warn(self, monkeypatch) -> None:
        import integrations.mcp_server._dispatch as dispatch_mod

        status = {
            "indexed_files": 1, "total_chunks": 1,
            "last_indexed": "2026-01-01T00:00:00Z",
            "embed_model": "test-model", "workspace_root": "/repo",
            "symbol_count": 0, "notes_count": 0, "languages": [],
            "grammars_unavailable": [], "stale_task_count": 1, "stale_task_oldest_id": 3,
        }
        # Default config (3) — 1 stale task must not render the line.
        assert "checkpoint(s)" not in self._call(status)

        # Lower the threshold to 1 — the same count of 1 must now render it.
        monkeypatch.setattr(dispatch_mod, "MEMORY_HYGIENE_STALE_TASK_WARN_COUNT", 1)
        assert "checkpoint(s)" in self._call(status)

    def test_line_age_text_reflects_config(self, monkeypatch) -> None:
        import integrations.mcp_server._dispatch as dispatch_mod

        monkeypatch.setattr(dispatch_mod, "MEMORY_HYGIENE_STALE_TASK_WARN_COUNT", 1)
        monkeypatch.setattr(dispatch_mod, "MEMORY_HYGIENE_STALE_TASK_WARN_AGE_DAYS", 30)
        status = {
            "indexed_files": 1, "total_chunks": 1,
            "last_indexed": "2026-01-01T00:00:00Z",
            "embed_model": "test-model", "workspace_root": "/repo",
            "symbol_count": 0, "notes_count": 0, "languages": [],
            "grammars_unavailable": [], "stale_task_count": 1, "stale_task_oldest_id": 3,
        }
        text = self._call(status)
        assert "30 days" in text


# ---------------------------------------------------------------------------
# REST-level: /v1/status
# ---------------------------------------------------------------------------

class TestRestStatusStaleTask:
    # UPG-CONFTEST-SERVICE-CLOBBER: the `client`/`client_real_memory` fixtures
    # now save/restore app.state.service in teardown, so the mock no longer
    # persists into this session-scoped real-service test — the former
    # `_reaffirm_real_service` defensive reassignment is no longer needed.

    def test_status_route_surfaces_stale_task_fields(self, real_service_client) -> None:
        client, svc, ws = real_service_client
        client.post("/v1/memory/clear", json={})

        resp = client.get("/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stale_task_count"] == 0
        assert data["stale_task_oldest_id"] is None

    def test_status_route_reflects_backdated_task_note(self, real_service_client) -> None:
        client, svc, ws = real_service_client
        client.post("/v1/memory/clear", json={})

        note_id = svc._context_store.remember(ws, "stale checkpoint", kind="task")
        _backdate(svc._context_store, note_id, age_days=10)

        resp = client.get("/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stale_task_count"] == 1
        assert data["stale_task_oldest_id"] == note_id

        # Clean up so later session-scoped tests aren't affected.
        client.post("/v1/memory/clear", json={})
