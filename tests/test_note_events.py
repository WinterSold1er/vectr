"""
Tests for the note lifecycle event log (UPG-MEMORY-STATE-MACHINE §4).

Two layers:
  - Pure `agent.working_context_store._events.fold()` — offline, no store,
    no I/O — covers tie-breaking, revert-of-revert, and the
    audit-only (stale_flagged/promoted) events that never change state.
  - `WorkingContextStore` integration — `contradicts=` on remember(),
    `revoke_note`/`reinstate_note`, `supersede_note` (post-hoc
    retirement, UPG-NOTE-RETIRE-POST-HOC), the migration writes for
    `supersedes`/`promote`, anti-memory rendering, proxy-anchor drift ->
    `stale_flagged`, and `forget()`'s cascade delete.
"""
from __future__ import annotations

import time

import pytest


# ---------------------------------------------------------------------------
# Pure fold() — offline
# ---------------------------------------------------------------------------

class TestFold:
    def test_empty_log_folds_active(self) -> None:
        from agent.working_context_store._events import fold
        result = fold([])
        assert result == {"state": "active", "reason": None, "actor": None, "ts": None}

    def test_created_alone_is_active(self) -> None:
        from agent.working_context_store._events import fold
        result = fold([{"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0}])
        assert result["state"] == "active"

    def test_revoked_after_created_is_revoked(self) -> None:
        from agent.working_context_store._events import fold
        events = [
            {"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0},
            {"id": 2, "event": "revoked", "actor": "agent", "reason": "wrong", "ts": 200.0},
        ]
        result = fold(events)
        assert result["state"] == "revoked"
        assert result["reason"] == "wrong"
        assert result["actor"] == "agent"
        assert result["ts"] == 200.0

    def test_reinstated_after_revoked_is_active_again(self) -> None:
        from agent.working_context_store._events import fold
        events = [
            {"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0},
            {"id": 2, "event": "revoked", "actor": "agent", "reason": "wrong", "ts": 200.0},
            {"id": 3, "event": "reinstated", "actor": "human", "reason": None, "ts": 300.0},
        ]
        result = fold(events)
        assert result["state"] == "active"
        assert result["actor"] == "human"

    def test_revoke_reinstate_revoke_again_lands_on_last_event(self) -> None:
        """Revert-of-revert is always legal and repeatable — the fold has no
        memory beyond the log itself, so a second revoke after a reinstate
        is indistinguishable from a first."""
        from agent.working_context_store._events import fold
        events = [
            {"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0},
            {"id": 2, "event": "revoked", "actor": "agent", "reason": "first reason", "ts": 200.0},
            {"id": 3, "event": "reinstated", "actor": "agent", "reason": None, "ts": 300.0},
            {"id": 4, "event": "revoked", "actor": "human", "reason": "second reason", "ts": 400.0},
        ]
        result = fold(events)
        assert result["state"] == "revoked"
        assert result["reason"] == "second reason"
        assert result["actor"] == "human"

    def test_stale_flagged_does_not_change_lifecycle_state(self) -> None:
        """stale_flagged is audit-only (§4.4 anchor drift) — orthogonal to
        the revoked/superseded/active lifecycle."""
        from agent.working_context_store._events import fold
        events = [
            {"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0},
            {"id": 2, "event": "stale_flagged", "actor": "system", "reason": None, "ts": 200.0},
        ]
        result = fold(events)
        assert result["state"] == "active"
        # the state-changing "created" event's actor/ts, not stale_flagged's
        assert result["actor"] == "agent"

    def test_promoted_does_not_change_lifecycle_state(self) -> None:
        from agent.working_context_store._events import fold
        events = [
            {"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0},
            {"id": 2, "event": "promoted", "actor": "human", "reason": None, "ts": 200.0, "payload": "human"},
        ]
        result = fold(events)
        assert result["state"] == "active"
        assert result["actor"] == "agent"

    def test_superseded_is_a_terminal_lifecycle_state(self) -> None:
        from agent.working_context_store._events import fold
        events = [
            {"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0},
            {"id": 2, "event": "superseded", "actor": "agent", "reason": None, "ts": 200.0, "payload": "7"},
        ]
        result = fold(events)
        assert result["state"] == "superseded"

    def test_tie_break_is_by_caller_supplied_ordering_not_ts(self) -> None:
        """fold() trusts the caller's ordering (must be sorted by monotonic
        `id`, per its docstring) rather than re-sorting by `ts` itself — two
        events can share an identical wall-clock `ts` (same call, same low-
        resolution tick) and the LAST one in iteration order still wins."""
        from agent.working_context_store._events import fold
        events = [
            {"id": 1, "event": "created", "actor": "agent", "reason": None, "ts": 100.0},
            {"id": 2, "event": "revoked", "actor": "agent", "reason": "r1", "ts": 100.0},
            {"id": 3, "event": "reinstated", "actor": "agent", "reason": None, "ts": 100.0},
        ]
        result = fold(events)
        assert result["state"] == "active"  # the LAST (id=3) event wins despite identical ts


# ---------------------------------------------------------------------------
# WorkingContextStore integration
# ---------------------------------------------------------------------------

def _store(tmp_path):
    from agent.working_context_store import WorkingContextStore
    return WorkingContextStore(str(tmp_path))


class TestContradicts:
    def test_contradicts_revokes_the_target_note(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "the API returns a list")
        new_id = store.remember(ws, "the API actually returns a dict", contradicts=old_id)
        states = store.note_event_states(ws, [store.get_note(ws, old_id), store.get_note(ws, new_id)])
        assert states[old_id]["state"] == "revoked"
        assert states[old_id]["actor"] == "agent"
        assert f"#{new_id}" in states[old_id]["reason"]
        # the new note itself is untouched (active)
        assert states.get(new_id, {"state": "active"})["state"] == "active"

    def test_contradicts_target_stays_live_for_recall(self, tmp_path) -> None:
        """Unlike `supersedes`, a revoked note is never excluded from
        recall — it must stay a live candidate so the anti-memory deterrent
        can surface (§4.3)."""
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old belief")
        store.remember(ws, "correction", contradicts=old_id)
        active = store.recall(ws)
        assert any(n.note_id == old_id for n in active)
        old = store.get_note(ws, old_id)
        assert old.valid_until is None  # never tombstoned, unlike supersedes

    def test_contradicts_nonexistent_note_rejected(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        with pytest.raises(ValueError):
            store.remember(ws, "a note", contradicts=999999)

    def test_contradicts_has_no_human_provenance_write_boundary_guard(self, tmp_path) -> None:
        """Deliberately distinct from `supersedes`: revoking a human note
        does not silently mute it (it stays visible in deterrent form), so
        an agent-provenance write MAY contradict a human-provenance note."""
        store, ws = _store(tmp_path), str(tmp_path)
        human_id = store.remember(ws, "a human finding", provenance="human")
        new_id = store.remember(ws, "an agent correction", contradicts=human_id, provenance="agent")
        states = store.note_event_states(ws, [store.get_note(ws, human_id)])
        assert states[human_id]["state"] == "revoked"
        assert new_id  # write succeeded


class TestRevokeReinstate:
    def test_revoke_note_appends_revoked_event(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        ok = store.revoke_note(ws, note_id, "turned out false", actor="human")
        assert ok is True
        states = store.note_event_states(ws, [store.get_note(ws, note_id)])
        assert states[note_id]["state"] == "revoked"
        assert states[note_id]["actor"] == "human"
        assert states[note_id]["reason"] == "turned out false"

    def test_revoke_note_does_not_hard_delete(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        store.revoke_note(ws, note_id, "wrong")
        assert store.get_note(ws, note_id) is not None
        active = store.recall(ws)
        assert any(n.note_id == note_id for n in active)

    def test_revoke_nonexistent_note_returns_false(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        assert store.revoke_note(ws, 999999, "reason") is False

    def test_revoke_rejects_system_actor(self, tmp_path) -> None:
        """`system` is reserved for the deterministic anchor-drift
        transition — a human/agent-initiated revoke call is always a
        judgment call, never that actor."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        with pytest.raises(ValueError):
            store.revoke_note(ws, note_id, "reason", actor="system")

    def test_revoke_rejects_unrecognised_actor(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        with pytest.raises(ValueError):
            store.revoke_note(ws, note_id, "reason", actor="not-a-real-actor")

    def test_reinstate_reverses_revoke(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        store.revoke_note(ws, note_id, "wrong")
        ok = store.reinstate_note(ws, note_id, actor="human", reason="verified correct after all")
        assert ok is True
        states = store.note_event_states(ws, [store.get_note(ws, note_id)])
        assert states[note_id]["state"] == "active"

    def test_reinstate_nonexistent_note_returns_false(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        assert store.reinstate_note(ws, 999999) is False

    def test_reinstate_rejects_system_actor(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        with pytest.raises(ValueError):
            store.reinstate_note(ws, note_id, actor="system")

    def test_reinstate_on_never_revoked_note_is_harmless(self, tmp_path) -> None:
        """Always legal, even as a no-op-in-spirit call — same 'just
        append, let the fold decide' philosophy as a repeat revoke."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        ok = store.reinstate_note(ws, note_id)
        assert ok is True
        states = store.note_event_states(ws, [store.get_note(ws, note_id)])
        assert states[note_id]["state"] == "active"


class TestPostHocSupersede:
    """UPG-NOTE-RETIRE-POST-HOC: supersession reached against an
    ALREADY-STORED note — the transition write-time `supersedes=` cannot
    reach (a replacement written without it, an old note only later
    discovered to be already-superseded-in-substance). Same lifecycle STATE
    as write-time `superseded`, but a distinct event KIND
    (`superseded_post_hoc`) so the audit trail records this was a post-hoc
    caller judgment rather than born of a write."""

    def _stale_rows(self, store, ws, note_id):
        with store._conn() as conn:
            return conn.execute(
                "SELECT event, actor, reason, payload FROM note_events "
                "WHERE workspace = ? AND note_id = ? ORDER BY id",
                (ws, note_id),
            ).fetchall()

    def test_supersede_appends_post_hoc_event_folding_to_superseded(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "the API returns a list")
        new_id = store.remember(ws, "the API returns a dict")
        ok = store.supersede_note(ws, note_id, superseded_by=new_id, reason="recorded without supersedes=", actor="agent")
        assert ok is True
        rows = self._stale_rows(store, ws, note_id)
        last = rows[-1]
        # Distinct KIND from the write-time path — machine-queryable honesty.
        assert last["event"] == "superseded_post_hoc"
        assert last["actor"] == "agent"
        assert last["reason"] == "recorded without supersedes="
        assert last["payload"] == str(new_id)
        states = store.note_event_states(ws, [store.get_note(ws, note_id)])
        # But the SAME lifecycle state as write-time supersession.
        assert states[note_id]["state"] == "superseded"

    def test_supersede_sets_the_same_tombstone_columns_as_write_time(self, tmp_path) -> None:
        """Both retirement paths land in the identical tombstone shape —
        same columns, same recall behaviour."""
        store, ws = _store(tmp_path), str(tmp_path)
        wt_old = store.remember(ws, "old checkpoint")
        store.remember(ws, "replacement checkpoint", supersedes=wt_old)
        ph_old = store.remember(ws, "another old note")
        ph_new = store.remember(ws, "its replacement")
        store.supersede_note(ws, ph_old, superseded_by=ph_new)
        for target in (wt_old, ph_old):
            note = store.get_note(ws, target)
            assert note.valid_until is not None
            assert note.superseded_at is not None
        assert store.get_note(ws, ph_old).superseded_by_note_id == ph_new
        # And the columns behave the same way recall cares about.
        active = store.recall(ws)
        assert all(n.note_id not in (wt_old, ph_old) for n in active)
        history_ids = {n.note_id for n in store.recall(ws, include_superseded=True)}
        assert {wt_old, ph_old} <= history_ids

    def test_retired_note_never_fires_again(self, tmp_path) -> None:
        """evaluate_note keys on valid_until, not the event log — the column
        write is load-bearing for exclusion from trigger firing too."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(
            ws, "a fired-at-boot note", kind="directive", triggers=[{"event": "session-start"}],
        )
        formatted, fired_before = store.fire_and_format(ws, event="session-start")
        assert note_id in fired_before
        store.supersede_note(ws, note_id, reason="moved on")
        _, fired_after = store.fire_and_format(ws, event="session-start")
        assert note_id not in fired_after

    def test_retired_note_renders_superseded_badge_never_revoked_deterrent(self, tmp_path) -> None:
        """The brief's rendering contract: a retired-by-supersession note
        must be distinguishable from a revoked one — factual badge, raw
        content intact, no 'Previously believed' framing."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "the deploy script lives in scripts/deploy.sh")
        new_id = store.remember(ws, "deploy moved to CI")
        store.supersede_note(ws, note_id, superseded_by=new_id, actor="human", reason="consolidated")
        rendered = store.format_notes_for_llm([store.get_note(ws, note_id)], detail="full")
        assert "[REVOKED]" not in rendered
        assert "Previously believed" not in rendered
        assert f"[superseded by @note#{new_id}" in rendered
        assert "the deploy script lives in scripts/deploy.sh" in rendered  # content survives, badged

    def test_supersede_without_successor_renders_unlinked_badge(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "orphaned claim")
        store.supersede_note(ws, note_id)
        rendered = store.format_notes_for_llm([store.get_note(ws, note_id)], detail="full")
        assert "[superseded," in rendered
        assert "[REVOKED]" not in rendered
        active = store.recall(ws)
        assert all(n.note_id != note_id for n in active)

    def test_supersede_nonexistent_note_returns_false(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        assert store.supersede_note(ws, 999999) is False

    def test_missing_successor_rejected_before_any_mutation(self, tmp_path) -> None:
        """Same half-applied-write guard as remember()'s supersedes path: a
        bad successor id must leave the target untouched."""
        import pytest
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a note")
        with pytest.raises(ValueError):
            store.supersede_note(ws, note_id, superseded_by=999999)
        note = store.get_note(ws, note_id)
        assert note.valid_until is None  # not tombstoned
        events = self._stale_rows(store, ws, note_id)
        assert [e["event"] for e in events] == ["created"]  # nothing appended

    def test_self_successor_rejected(self, tmp_path) -> None:
        import pytest
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a note")
        with pytest.raises(ValueError):
            store.supersede_note(ws, note_id, superseded_by=note_id)

    def test_supersede_rejects_system_actor(self, tmp_path) -> None:
        """`system` is reserved for the deterministic transitions — deciding
        a stored note is already-superseded is always a judgment call."""
        import pytest
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        with pytest.raises(ValueError):
            store.supersede_note(ws, note_id, actor="system")

    def test_human_provenance_write_boundary_guard(self, tmp_path) -> None:
        """Same inversion hole remember()'s supersedes guard closes: an
        agent-actor call may not silence a human-provenance note; only a
        human-actor call may."""
        import pytest
        store, ws = _store(tmp_path), str(tmp_path)
        human_id = store.remember(ws, "a human-recorded rule", provenance="human")
        with pytest.raises(ValueError):
            store.supersede_note(ws, human_id, actor="agent")
        note = store.get_note(ws, human_id)
        assert note.valid_until is None
        assert store.supersede_note(ws, human_id, actor="human") is True

    def test_reinstate_reverses_a_post_hoc_supersession(self, tmp_path) -> None:
        """THE reversal contract: retire then reinstate must put the note
        back into default recall with its tombstone columns cleared — one
        appended event, never a rewrite of the retirement."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "wrongly retired")
        new_id = store.remember(ws, "its replacement")
        store.supersede_note(ws, note_id, superseded_by=new_id)
        assert all(n.note_id != note_id for n in store.recall(ws))
        assert store.reinstate_note(ws, note_id, reason="retirement was wrong") is True
        note = store.get_note(ws, note_id)
        assert note.valid_until is None
        assert note.superseded_at is None
        assert note.superseded_by_note_id is None
        assert any(n.note_id == note_id for n in store.recall(ws))
        states = store.note_event_states(ws, [note])
        assert states[note_id]["state"] == "active"

    def test_reinstate_reverses_a_write_time_supersession_too(self, tmp_path) -> None:
        """Pre-existing gap the post-hoc work closes: reinstate's own docstring
        always promised 'return to active regardless of which terminal state
        preceded it', but a write-time-superseded note stayed excluded from
        default recall because its valid_until column was never cleared."""

        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old checkpoint")
        store.remember(ws, "new checkpoint", supersedes=old_id)
        assert all(n.note_id != old_id for n in store.recall(ws))
        store.reinstate_note(ws, old_id)
        assert any(n.note_id == old_id for n in store.recall(ws))
        assert store.get_note(ws, old_id).valid_until is None


class TestSupersedesPromoteMigration:
    def test_supersedes_writes_a_superseded_event(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        old_id = store.remember(ws, "old finding")
        new_id = store.remember(ws, "corrected finding", supersedes=old_id)
        real_old = store.get_note(ws, old_id)
        states = store.note_event_states(ws, [real_old])
        assert states[old_id]["state"] == "superseded"
        assert states[old_id]["reason"] is None  # payload carries the new id, not reason
        # existing valid_until/superseded_by_note_id behaviour is unchanged
        assert real_old.valid_until is not None
        assert real_old.superseded_by_note_id == new_id

    def test_promote_writes_a_promoted_event_without_changing_lifecycle_state(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding", provenance="auto")
        store.promote(ws, note_id, "agent")
        note = store.get_note(ws, note_id)
        states = store.note_event_states(ws, [note])
        # promoted is audit-only — state stays "active", not some new value
        assert states.get(note_id, {"state": "active"})["state"] == "active"
        assert note.provenance == "agent"  # pre-existing promotion behaviour unchanged


class TestAntiMemoryRendering:
    def test_revoked_note_renders_deterrent_not_raw_content(self, tmp_path) -> None:
        """The anti-memory block substitutes a ONE-LINE summary (§4.3) for
        the note's raw content — a multi-line body's later lines must never
        render again once revoked, only the deterrent framing + a short
        summary + the revocation reason."""
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(
            ws,
            "the config lives in settings.py\n"
            "detail line that must never re-render once revoked\n"
            "another body line that must never re-render once revoked",
        )
        store.revoke_note(ws, note_id, "actually it moved to config.py")
        note = store.get_note(ws, note_id)
        rendered = store.format_notes_for_llm([note], detail="full")
        assert "detail line that must never re-render once revoked" not in rendered
        assert "another body line that must never re-render once revoked" not in rendered
        assert "Previously believed" in rendered
        assert "actually it moved to config.py" in rendered
        assert "[REVOKED]" in rendered

    def test_reinstated_note_renders_raw_content_again(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding that gets reinstated")
        store.revoke_note(ws, note_id, "temporarily doubted")
        store.reinstate_note(ws, note_id)
        note = store.get_note(ws, note_id)
        rendered = store.format_notes_for_llm([note], detail="full")
        assert "a finding that gets reinstated" in rendered
        assert "Previously believed" not in rendered

    def test_active_note_unaffected_by_anti_memory_rendering(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "an ordinary active finding")
        note = store.get_note(ws, note_id)
        rendered = store.format_notes_for_llm([note], detail="full")
        assert "an ordinary active finding" in rendered
        assert "[REVOKED]" not in rendered

    def test_revoked_note_index_line_carries_revoked_marker(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        store.revoke_note(ws, note_id, "wrong")
        note = store.get_note(ws, note_id)
        rendered = store.format_notes_for_llm([note], detail="index")
        assert "[REVOKED]" in rendered

    def test_revoked_note_still_appears_in_fire_and_format(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a directive-like finding", kind="directive", provenance="agent")
        store.revoke_note(ws, note_id, "no longer true")
        formatted, fired_ids = store.fire_and_format(ws, event="session-start")
        assert note_id in fired_ids
        assert "Previously believed" in formatted

    def test_revoked_note_deterrent_warning_survives_hook_channel_trim(self, tmp_path) -> None:
        """UPG-DETERRENT-TITLE-ONLY: agent/trigger_engine.py's
        `_trim_full_text_to_cap()` (the hook full-tier channel's own
        per-kind cap trim, e.g. gotcha's 100-token/400-char cap) truncates
        the raw `_format_full_block()` text with no template awareness at
        all -- unlike the proxy channel, which used to reorder the
        deterrent clause itself at render time. A long free-text
        revocation `reason` (unbounded, no cap of its own) could previously
        push the trim's right-truncation past the entire "Previously
        believed (...)" clause and lose the "Do not re-derive..." warning
        entirely -- exactly the silent-caveat-loss the design doc's
        poisoning-mitigation row forbids ("visible caveat, never silent
        suppression"). `_ANTI_MEMORY_TEMPLATE` now puts the warning FIRST,
        so it survives any right-truncation as long as the cap fits the
        warning sentence alone (~65 chars, far under every configured
        per-kind cap)."""
        store, ws = _store(tmp_path), str(tmp_path)
        long_reason = "this note was wrong because " + "the underlying assumption changed " * 12
        note_id = store.remember(
            ws, "a gotcha about retry timeouts", kind="gotcha",
            triggers=[{"event": "session-start"}],
        )
        store.revoke_note(ws, note_id, long_reason)
        formatted, fired_ids = store.fire_and_format(ws, event="session-start")
        assert note_id in fired_ids
        assert "Do not re-derive this from other sources without verification." in formatted


class TestProxyAnchorDrift:
    def test_anchor_drift_appends_stale_flagged_event(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("original lock content")
        note_id = store.remember(
            ws, "the lockfile pins numpy 1.26", kind="reference", anchors=["requirements.lock"],
        )
        f.write_text("changed lock content")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        note = store.get_note(ws, note_id)
        states = store.note_event_states(ws, [note])
        assert states[note_id]["state"] == "active"  # audit-only, not a lifecycle change

        with store._conn() as conn:
            rows = conn.execute(
                "SELECT event, actor FROM note_events WHERE workspace = ? AND note_id = ? "
                "AND event = 'stale_flagged'",
                (ws, note_id),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["actor"] == "system"

    def test_anchor_drift_verdict_line_in_rendering(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "Dockerfile"
        f.write_text("FROM python:3.11")
        note_id = store.remember(
            ws, "build process pinned in the Dockerfile", kind="reference", anchors=["Dockerfile"],
        )
        f.write_text("FROM python:3.12")
        notes = store.recall(ws)
        stale = store.check_staleness(notes, ws)
        rendered = store.format_notes_for_llm(notes, stale_warnings=stale, detail="full")
        assert "VERDICT: anchor changed since" in rendered

    def test_repeated_check_staleness_does_not_duplicate_stale_flagged_events(self, tmp_path) -> None:
        """Edge-triggered, not level-triggered: the event fires once on the
        transition into drift, not on every subsequent check while the
        drift persists."""
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "poetry.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["poetry.lock"])
        f.write_text("v2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        store.check_staleness(notes, ws)
        store.check_staleness(notes, ws)
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS n FROM note_events WHERE workspace = ? AND note_id = ? "
                "AND event = 'stale_flagged'",
                (ws, note_id),
            ).fetchall()
        assert rows[0]["n"] == 1

    def test_unchanged_anchor_never_flagged_stale(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "Pipfile.lock"
        f.write_text("stable")
        note_id = store.remember(ws, "pinned deps", kind="reference", anchors=["Pipfile.lock"])
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS n FROM note_events WHERE workspace = ? AND note_id = ? "
                "AND event = 'stale_flagged'",
                (ws, note_id),
            ).fetchall()
        assert rows[0]["n"] == 0


class TestStaleFlagDriftSignature:
    """UPG-STALE-FLAG-DOUBLE-EVENT: `_flag_stale_anchors()` suppresses a
    duplicate `stale_flagged` audit row by comparing the note's CURRENT
    drift signature (a sha256[:16] of its drifted anchors' sorted
    `path:current_hash` pairs) against the payload of its own MOST RECENT
    `stale_flagged` event — never against the most recent event of ANY
    kind. This closes two gaps in the older edge-trigger: an intervening
    lifecycle event (revoke/reinstate/promote) no longer admits a
    duplicate row, and a genuinely NEW drift of the same anchor is no
    longer permanently suppressed by an old signature."""

    def _stale_rows(self, store, ws, note_id):
        with store._conn() as conn:
            return conn.execute(
                "SELECT payload FROM note_events WHERE workspace = ? AND note_id = ? "
                "AND event = 'stale_flagged' ORDER BY id",
                (ws, note_id),
            ).fetchall()

    def test_drift_detected_once_writes_one_row_with_a_payload(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["requirements.lock"])
        f.write_text("v2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 1
        assert rows[0]["payload"] is not None

    def test_repeated_checks_of_unchanged_drift_stay_at_one_row(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["requirements.lock"])
        f.write_text("v2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        store.check_staleness(notes, ws)
        store.check_staleness(notes, ws)
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 1

    def test_intervening_lifecycle_event_does_not_admit_a_duplicate_row(self, tmp_path) -> None:
        """THE REGRESSION this task fixes: revoking then reinstating the
        note between two checks of the exact same unchanged drift must not
        cause a second `stale_flagged` row. Against the old "most recent
        event of ANY kind" implementation this fails, because the most
        recent event becomes `reinstated`, not `stale_flagged`, so the old
        suppression check no longer sees a match and re-flags."""
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["requirements.lock"])
        f.write_text("v2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        store.revoke_note(ws, note_id, "double-checking, unrelated to the drift")
        store.reinstate_note(ws, note_id)
        store.check_staleness(notes, ws)
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 1

    def test_intervening_promote_does_not_admit_a_duplicate_row(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(
            ws, "lockfile pin", kind="reference", anchors=["requirements.lock"], provenance="auto",
        )
        f.write_text("v2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        store.promote(ws, note_id, "agent")
        store.check_staleness(notes, ws)
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 1

    def test_new_drift_of_the_same_anchor_writes_a_second_distinct_row(self, tmp_path) -> None:
        """The previously-missed gap: a SECOND, genuinely different drift of
        the same anchor must write its own fresh event, not be silently
        swallowed forever by the first stale_flagged row."""
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["requirements.lock"])
        f.write_text("v2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        f.write_text("v3")
        store.check_staleness(notes, ws)
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 2
        assert rows[0]["payload"] != rows[1]["payload"]
        assert rows[0]["payload"] is not None and rows[1]["payload"] is not None

    def test_anchor_restored_reports_no_new_row_and_stops_appearing(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["requirements.lock"])
        f.write_text("v2")
        notes = store.recall(ws)
        stale = store.check_staleness(notes, ws)
        assert note_id in stale
        f.write_text("v1")  # restored to write-time content
        stale_again = store.check_staleness(notes, ws)
        assert note_id not in stale_again
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 1  # restoring the anchor writes no event of its own

    def test_premigration_null_payload_row_gets_one_resync_then_settles(self, tmp_path) -> None:
        """Backward compatibility: a `stale_flagged` row written before this
        signature scheme existed carries `payload = NULL`. `None` never
        equals a real signature, so the first check after upgrade appends
        exactly one fresh event carrying the current signature, and steady
        state (no further rows for the same unchanged drift) resumes
        immediately."""
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["requirements.lock"])
        f.write_text("v2")
        with store._conn() as conn:
            conn.execute(
                "INSERT INTO note_events (workspace, note_id, event, actor, reason, payload, ts) "
                "VALUES (?, ?, 'stale_flagged', 'system', NULL, NULL, ?)",
                (ws, note_id, time.time()),
            )
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 2
        assert rows[0]["payload"] is None
        assert rows[1]["payload"] is not None
        store.check_staleness(notes, ws)
        rows_after = self._stale_rows(store, ws, note_id)
        assert len(rows_after) == 2  # steady state resumed, no third row

    def test_signature_is_order_independent_and_set_specific(self, tmp_path) -> None:
        """Two notes whose drifted anchor sets are the SAME paths+hashes
        must produce the SAME signature (order-independent — anchors are
        declared in whatever order the caller listed them); a note with a
        genuinely different drifted anchor set must produce a different
        one."""
        store, ws = _store(tmp_path), str(tmp_path)
        a = tmp_path / "a.lock"
        b = tmp_path / "b.lock"
        a.write_text("a1")
        b.write_text("b1")
        note_same_order = store.remember(
            ws, "two-anchor note", kind="reference", anchors=["a.lock", "b.lock"],
        )
        note_reverse_order = store.remember(
            ws, "same two anchors, declared in reverse", kind="reference", anchors=["b.lock", "a.lock"],
        )
        note_different_set = store.remember(
            ws, "only one of the two anchors", kind="reference", anchors=["a.lock"],
        )
        a.write_text("a2")
        b.write_text("b2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        rows_same = self._stale_rows(store, ws, note_same_order)
        rows_reverse = self._stale_rows(store, ws, note_reverse_order)
        rows_different = self._stale_rows(store, ws, note_different_set)
        assert rows_same[0]["payload"] == rows_reverse[0]["payload"]
        assert rows_same[0]["payload"] != rows_different[0]["payload"]

    def test_fold_lifecycle_state_unaffected_by_any_number_of_stale_flagged_rows(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        f = tmp_path / "requirements.lock"
        f.write_text("v1")
        note_id = store.remember(ws, "lockfile pin", kind="reference", anchors=["requirements.lock"])
        f.write_text("v2")
        notes = store.recall(ws)
        store.check_staleness(notes, ws)
        f.write_text("v3")
        store.check_staleness(notes, ws)
        note = store.get_note(ws, note_id)
        states = store.note_event_states(ws, [note])
        assert states[note_id]["state"] == "active"

        store.revoke_note(ws, note_id, "unrelated to the drift")
        f.write_text("v4")
        notes_after_revoke = store.recall(ws)
        store.check_staleness(notes_after_revoke, ws)
        note = store.get_note(ws, note_id)
        states = store.note_event_states(ws, [note])
        assert states[note_id]["state"] == "revoked"
        rows = self._stale_rows(store, ws, note_id)
        assert len(rows) == 3  # three distinct drifts, unaffected by the revoke in between


class TestForgetCascade:
    def test_forget_deletes_note_events(self, tmp_path) -> None:
        store, ws = _store(tmp_path), str(tmp_path)
        note_id = store.remember(ws, "a finding")
        store.revoke_note(ws, note_id, "wrong")
        store.forget(ws, note_id)
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS n FROM note_events WHERE workspace = ? AND note_id = ?",
                (ws, note_id),
            ).fetchall()
        assert rows[0]["n"] == 0
