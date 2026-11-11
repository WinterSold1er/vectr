"""Service-level proactive + artifact-cache tests (UPG-PRO-1/4/5/7 + caching).

Uses a memory-only VectrService with the dummy embedder — no reranker, no model
download, no network — so the wiring (matcher -> gate, scored recall, cache
invalidation) is exercised end to end without the heavy full-mode stack.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.searcher import SearchResult, SearchResultList


def _service(tmp_path, monkeypatch, proactive_now_fn=None, **env):
    from agent import indexer as idx_module
    from tests.conftest import _DummyEmbedProvider

    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with patch("integrations.vscode_bridge.configure_all"), \
         patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
         patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp_path / "db")}):
        from app.service import VectrService
        return VectrService(
            workspace_root=str(tmp_path), memory_only=True, proactive_now_fn=proactive_now_fn,
        )


# -- proactive_context ------------------------------------------------------

def test_proactive_disabled_returns_empty_for_hook_channel(tmp_path, monkeypatch):
    # The master opt-in still gates AMBIENT surfaces: hook-channel requests
    # inject nothing while proactive.enabled is explicitly off. (Since
    # UPG-PROXY-DEFAULT-ON-GATE, proactive.enabled defaults ON, so this must
    # set VECTR_PROACTIVE=0 explicitly rather than relying on it being unset.)
    monkeypatch.setenv("VECTR_PROACTIVE", "0")
    svc = _service(tmp_path, monkeypatch)
    svc.remember("resolver.py: the lock drops on scope exit", kind="gotcha")
    out = svc.proactive_context(text="lock", file_paths=["/x/resolver.py"], channel="hook")
    assert out == {"context": "", "item_count": 0, "anchor_ids": [], "scores": []}


def test_proactive_proxy_channel_injects_without_master_switch(tmp_path, monkeypatch):
    # UPG-PROXY-HIDDEN-MASTER-SWITCH: launching the proxy IS the consent for
    # the proxy channel — it must inject even with the master opt-in
    # explicitly off. (Since UPG-PROXY-DEFAULT-ON-GATE, proactive.enabled
    # defaults ON, so this must set VECTR_PROACTIVE=0 explicitly to exercise
    # the "master switch off" case this test is named for.)
    monkeypatch.setenv("VECTR_PROACTIVE", "0")
    svc = _service(tmp_path, monkeypatch)
    nid = svc.remember("resolver.py holds the workspace lock; drops on scope exit", kind="gotcha")
    out = svc.proactive_context(text="", file_paths=["/abs/resolver.py"], session_id="s1", channel="proxy")
    assert out["item_count"] >= 1
    assert f"note:{nid}" in out["anchor_ids"]
    assert svc.get_proactive_injection_counts().get("proxy", 0) == 1


def test_status_exposes_proactive_enabled(tmp_path, monkeypatch):
    # UPG-PROXY-DEFAULT-ON-GATE: the master opt-in now defaults ON when
    # VECTR_PROACTIVE is unset, and the env var still turns it OFF.
    monkeypatch.delenv("VECTR_PROACTIVE", raising=False)
    svc = _service(tmp_path, monkeypatch)
    assert svc.status()["proactive_enabled"] is True
    monkeypatch.setenv("VECTR_PROACTIVE", "0")
    assert svc.status()["proactive_enabled"] is False


def test_status_proactive_enabled_reflects_bind_gate(tmp_path, monkeypatch):
    # UPG-PROXY-STATUS-TRUE-STATE: on a non-loopback bind proactive_context()
    # refuses every channel unconditionally, so status must not report the
    # bare config value (post-flip: True) where injection is in fact refused.
    monkeypatch.delenv("VECTR_PROACTIVE", raising=False)
    svc = _service(tmp_path, monkeypatch)
    monkeypatch.setenv("VECTR_BIND_HOST", "0.0.0.0")
    assert svc.status()["proactive_enabled"] is False
    monkeypatch.setenv("VECTR_BIND_HOST", "127.0.0.1")
    assert svc.status()["proactive_enabled"] is True


def test_proactive_structural_note_injected(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    nid = svc.remember("resolver.py holds the workspace lock; drops on scope exit", kind="gotcha")
    out = svc.proactive_context(text="", file_paths=["/abs/resolver.py"], session_id="s1")
    assert out["item_count"] >= 1
    assert f"note:{nid}" in out["anchor_ids"]
    assert "resolver.py" in out["context"]
    # A metadata-only injection was counted.
    assert svc.get_proactive_injection_counts().get("proxy", 0) == 1


def test_proactive_no_match_returns_empty(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    svc.remember("resolver.py note", kind="gotcha")
    out = svc.proactive_context(text="", file_paths=["/x/unrelated_file.py"], session_id="s1")
    assert out["item_count"] == 0
    assert out["context"] == ""


def test_proactive_dedup_cooldown_across_calls(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    svc.remember("resolver.py holds the lock", kind="gotcha")
    first = svc.proactive_context(file_paths=["/x/resolver.py"], session_id="sess")
    second = svc.proactive_context(file_paths=["/x/resolver.py"], session_id="sess")
    assert first["item_count"] == 1
    assert second["item_count"] == 0  # cooldown suppresses the repeat


def test_proactive_cooldown_ttl_expires_via_injected_clock(tmp_path, monkeypatch):
    """UPG-GATE-NOWFN-PLUMBING end to end at service level: the clock
    injected into VectrService reaches the proxy channel's cooldown
    LedgerStore (SessionLedger's now_fn seam, previously unreachable from
    above app/service.py), so TTL expiry is provable without sleeping.

    Frozen inside the TTL window, a repeat delivery stays suppressed; once
    the injected clock passes cooldown_ttl_seconds the same note is
    re-admitted. If the plumbing regresses and the service builds its
    LedgerStore without now_fn, real time.monotonic never advances between
    the calls and the final call still sees the suppression — the last
    assertion fails."""
    clock = {"t": 1000.0}
    svc = _service(
        tmp_path, monkeypatch,
        VECTR_PROACTIVE="1",
        VECTR_PROACTIVE_COOLDOWN_TTL_SECONDS="3600",
        proactive_now_fn=lambda: clock["t"],
    )
    svc.remember("resolver.py holds the lock", kind="gotcha")
    first = svc.proactive_context(file_paths=["/x/resolver.py"], session_id="sess")
    assert first["item_count"] == 1
    # Direct seam pin (ledger is built lazily on the first gate pass): OUR
    # clock is what the ledger reads, not time.monotonic.
    assert svc._proactive_ledger is not None
    assert svc._proactive_ledger._now_fn() == 1000.0

    second = svc.proactive_context(file_paths=["/x/resolver.py"], session_id="sess")
    assert second["item_count"] == 0  # cooldown suppresses within the TTL...

    clock["t"] += 3600.0  # ...and stops suppressing once the TTL has elapsed
    third = svc.proactive_context(file_paths=["/x/resolver.py"], session_id="sess")
    assert third["item_count"] == 1


def test_proactive_declared_anchor_note_survives_reads_and_retires_on_edit(tmp_path, monkeypatch):
    """UPG-PROXY-INJECT-SINGLE-TURN end to end: a declared-anchor note keeps
    resurfacing across requests that only READ its anchored file (unlike
    test_proactive_dedup_cooldown_across_calls's plain content-mention note,
    which retires after one delivery), and only retires once a request's
    `edited_file_paths` shows the file actually being edited.

    Writes a real file under the workspace (same pattern as
    test_proxy_and_hook_channels_share_one_turn_ledger) so the note's
    declared anchor "resolver.py" resolves to the request's absolute path
    via `_path_trigger_candidates()`'s workspace-relative form -- an
    anchor recorded exactly this way is `recall_for_path`'s strongest
    signal (`_anchors_exact_match`), independent of the note's prose."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    (tmp_path / "resolver.py").write_text("# lock resolver\n")
    abs_path = str(tmp_path / "resolver.py")
    nid = svc.remember(
        "the retry limit here is 3, do not raise it without checking the caller",
        kind="gotcha", anchors=["resolver.py"],
    )
    read_only = dict(file_paths=[abs_path], session_id="sess")

    first = svc.proactive_context(**read_only)
    second = svc.proactive_context(**read_only)
    assert first["item_count"] == 1 and f"note:{nid}" in first["anchor_ids"]
    assert second["item_count"] == 1 and f"note:{nid}" in second["anchor_ids"]

    edited = svc.proactive_context(
        file_paths=[abs_path], edited_file_paths=[abs_path], session_id="sess",
    )
    assert edited["item_count"] == 1 and f"note:{nid}" in edited["anchor_ids"]

    after_edit = svc.proactive_context(**read_only)
    assert after_edit["item_count"] == 0  # retired: the decision already happened


# -- loopback bind enforcement (UPG-PROXY-LOOPBACK-BYPASS) ------------------

def test_proactive_refuses_non_loopback_bind_on_proxy_channel(tmp_path, monkeypatch):
    # B4: channel="proxy" is the field's DEFAULT and was exempt from BOTH the
    # master switch and the bind check. The bind refusal must be
    # unconditional and channel-independent: it fires even for the proxy
    # channel's own launch-is-consent exemption, and even with the master
    # switch on.
    monkeypatch.setenv("VECTR_BIND_HOST", "0.0.0.0")
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    svc.remember("resolver.py holds the workspace lock; drops on scope exit", kind="gotcha")
    out = svc.proactive_context(text="", file_paths=["/abs/resolver.py"], session_id="s1", channel="proxy")
    assert out == {"context": "", "item_count": 0, "anchor_ids": [], "scores": []}


def test_proactive_refuses_non_loopback_bind_regardless_of_channel(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTR_BIND_HOST", "10.0.0.5")
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    svc.remember("resolver.py holds the workspace lock; drops on scope exit", kind="gotcha")
    for channel in ("proxy", "hook", "SessionStart", "anything-a-client-sends"):
        out = svc.proactive_context(
            text="", file_paths=["/abs/resolver.py"], session_id="s1", channel=channel
        )
        assert out["item_count"] == 0, channel


def test_proactive_still_injects_on_loopback_bind(tmp_path, monkeypatch):
    # Regression guard: the new bind check must not break the ordinary,
    # default (loopback) posture — VECTR_BIND_HOST unset falls back to
    # 127.0.0.1, which is loopback.
    monkeypatch.delenv("VECTR_BIND_HOST", raising=False)
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    nid = svc.remember("resolver.py holds the workspace lock; drops on scope exit", kind="gotcha")
    out = svc.proactive_context(text="", file_paths=["/abs/resolver.py"], session_id="s1", channel="proxy")
    assert f"note:{nid}" in out["anchor_ids"]


# -- cross-channel dedup (UPG-PROXY-CROSS-CHANNEL-DEDUP) --------------------

def test_proxy_and_hook_channels_share_one_turn_ledger(tmp_path, monkeypatch):
    """Integration proof (service level) that `proactive_context` (proxy)
    and `recall`/`_recall_impl` (hook/trigger-engine surface) consult the
    SAME per-session `TurnInjectionLedger` via `_turn_ledger_for` — the same
    note must not inject twice across both channels in one turn under a
    shared session id."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    (tmp_path / "resolver.py").write_text("# lock resolver\n")
    abs_path = str(tmp_path / "resolver.py")
    nid = svc.remember(
        "resolver.py holds the workspace lock; drops on scope exit",
        kind="gotcha",
        anchors=["resolver.py"],
    )
    sid = "shared-session"

    # Surface 1 (hook/trigger-engine): PreToolUse file-anchored delivery
    # fires the note's declared pre-edit trigger and claims it.
    hook_text = svc.recall(file_path=abs_path, session_id=sid, hook_event="PreToolUse")
    assert "resolver.py" in hook_text

    # Surface 2 (proxy channel), SAME session/turn: already claimed by
    # surface 1 this turn -> must NOT re-inject.
    proxy_out = svc.proactive_context(
        text="", file_paths=[abs_path], session_id=sid, channel="proxy",
    )
    assert f"note:{nid}" not in proxy_out["anchor_ids"]

    # Non-vacuous: a FRESH session's proxy call (ledger untouched) DOES
    # inject the same note.
    fresh_out = svc.proactive_context(
        text="", file_paths=[abs_path], session_id="fresh-session", channel="proxy",
    )
    assert f"note:{nid}" in fresh_out["anchor_ids"]

    # After the shared session's turn boundary (a real UserPromptSubmit),
    # the proxy channel becomes eligible again.
    svc.reset_turn_ledger(sid)
    reset_out = svc.proactive_context(
        text="", file_paths=[abs_path], session_id=sid, channel="proxy",
    )
    assert f"note:{nid}" in reset_out["anchor_ids"]


def test_proxy_call_with_unseen_session_id_does_not_allocate_turn_ledger(tmp_path, monkeypatch):
    """`proactive_context` must consult the shared TurnInjectionLedger
    LOOKUP-ONLY (`_existing_turn_ledger`), never allocate one
    (`_turn_ledger_for`). The proxy channel's session_id (a per-proxy-
    process instance id minted once by `VectrProxy.__init__`, see
    UPG-PROXY-COOLDOWN-SESSION-IDENTITY) is, by construction, disjoint from
    the editor-supplied session id hook/trigger-engine surfaces use — if a
    proxy call allocated a ledger no hook surface will ever look up, that
    churn would consume slots in the EVICTION_MAX_TRACKED_SESSIONS-bounded
    `_turn_ledgers` LRU for zero dedup benefit, risking eviction of a live
    editor session's real ledger under heavy multi-subagent session churn."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    svc.remember("resolver.py holds the workspace lock; drops on scope exit", kind="gotcha")
    assert len(svc._turn_ledgers) == 0

    svc.proactive_context(
        text="", file_paths=["/abs/resolver.py"], session_id="never-seen-before", channel="proxy",
    )
    assert len(svc._turn_ledgers) == 0
    assert "never-seen-before" not in svc._turn_ledgers


def test_proxy_instance_shaped_session_id_does_not_evict_editor_turn_ledger(tmp_path, monkeypatch):
    """Turn-ledger non-regression (UPG-PROXY-COOLDOWN-SESSION-IDENTITY test
    C). A proxy-channel call whose session_id is shaped exactly like
    `VectrProxy._instance_id` (`proxy-<16 hex chars>`) must still hit the
    LOOKUP-ONLY `_existing_turn_ledger` path, never `_turn_ledger_for`: no
    new entry is allocated in `_turn_ledgers`, and a real editor-session
    ledger allocated beforehand by a hook surface survives untouched (same
    object, not just an equal one)."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    (tmp_path / "resolver.py").write_text("# lock resolver\n")
    abs_path = str(tmp_path / "resolver.py")
    svc.remember(
        "resolver.py holds the workspace lock; drops on scope exit",
        kind="gotcha", anchors=["resolver.py"],
    )
    editor_sid = "editor-session-abc"
    svc.recall(file_path=abs_path, session_id=editor_sid, hook_event="PreToolUse")
    assert editor_sid in svc._turn_ledgers
    editor_ledger_before = svc._turn_ledgers[editor_sid]
    len_before = len(svc._turn_ledgers)

    proxy_sid = "proxy-" + "a" * 16  # shaped exactly like VectrProxy._instance_id
    svc.proactive_context(
        text="", file_paths=[abs_path], session_id=proxy_sid, channel="proxy",
    )

    assert len(svc._turn_ledgers) == len_before
    assert proxy_sid not in svc._turn_ledgers
    assert editor_sid in svc._turn_ledgers
    assert svc._turn_ledgers[editor_sid] is editor_ledger_before


def test_proactive_revoked_note_renders_deterrent_not_raw_content(tmp_path, monkeypatch):
    """UPG-PROXY-REVOKED-LEAK end-to-end: proactive_context() must render a
    revoked note's anti-memory deterrent -- never its raw content as active
    fact -- through the FULL service stack (store -> recall_for_path() ->
    _ServiceMatchSource.note_states() -> ProactiveMatcher -> gate -> context
    string). Non-vacuity: the same note, while still active, injects its raw
    content verbatim first, proving the match itself is real and not an
    artifact of the revoked-state rendering path."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    nid = svc.remember(
        "resolver.py holds the workspace lock; drops on scope exit", kind="gotcha"
    )
    active = svc.proactive_context(
        text="", file_paths=["/abs/resolver.py"], session_id="s1"
    )
    assert active["item_count"] >= 1
    assert "workspace lock" in active["context"]

    svc.revoke_note(nid, reason="wrong assumption")

    # A different session_id: the dedup cooldown suppresses a repeat
    # injection for the SAME anchor within a session regardless of this fix,
    # so reusing "s1" here would produce a false pass/fail either way.
    revoked = svc.proactive_context(
        text="", file_paths=["/abs/resolver.py"], session_id="s2"
    )
    assert revoked["item_count"] >= 1
    assert f"note:{nid}" in revoked["anchor_ids"]
    assert "REVOKED" in revoked["context"]
    assert "Do not re-derive" in revoked["context"]


# -- structural-channel precision (UPG-PROXY-INJECT-PRECISION) --------------

def test_proactive_task_kind_note_excluded_from_structural_channel(tmp_path, monkeypatch):
    """Lever 1: `_ServiceMatchSource.structural_notes()` filters on
    `note.kind` (a static, config-declared allowlist -- never query
    content). A kind="task" note anchored to the exact window file is
    excluded end to end, while a kind="gotcha" note anchored to the same
    file still injects -- proving the filter is real, not vacuous."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    task_nid = svc.remember(
        "still investigating the resolver.py retry path", kind="task",
        anchors=["resolver.py"],
    )
    gotcha_nid = svc.remember(
        "resolver.py holds the workspace lock; drops on scope exit", kind="gotcha",
        anchors=["resolver.py"],
    )
    out = svc.proactive_context(
        text="", file_paths=["/abs/resolver.py"], session_id="s1",
    )
    assert f"note:{gotcha_nid}" in out["anchor_ids"]
    assert f"note:{task_nid}" not in out["anchor_ids"]


def test_proactive_kind_filtered_note_never_claimed_in_turn_ledger(tmp_path, monkeypatch):
    """Ledger rule pin, end to end through the real service +
    TurnInjectionLedger (UPG-PROXY-INJECT-PRECISION ledger rule): a note
    excluded by lever 1's kind filter never becomes a `Candidate`, so it is
    never eligible to be claimed -- confirmed here against the actual
    per-session ledger the proxy/hook surfaces share, not a stand-in."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    task_nid = svc.remember(
        "still investigating the resolver.py retry path", kind="task",
        anchors=["resolver.py"],
    )
    gotcha_nid = svc.remember(
        "resolver.py holds the workspace lock; drops on scope exit", kind="gotcha",
        anchors=["resolver.py"],
    )
    sid = "s-ledger-pin"
    turn_ledger = svc._turn_ledger_for(sid)  # pre-allocate so proactive_context
                                              # (lookup-only) can find it.
    out = svc.proactive_context(
        text="", file_paths=["/abs/resolver.py"], session_id=sid,
    )
    assert f"note:{gotcha_nid}" in out["anchor_ids"]
    assert turn_ledger.eligible(gotcha_nid) is False  # delivered: claimed
    assert turn_ledger.eligible(task_nid) is True      # kind-filtered: NEVER claimed


def test_proactive_directive_kind_note_excluded_from_structural_channel(tmp_path, monkeypatch):
    """Lever 1's `structural_kinds` allowlist excludes kind="directive" for
    the same reason it excludes "task" (config.yaml's `proactive.
    structural_kinds` comment: "not a durable fact about the file"). This is
    a distinct, independent mechanism from the sibling `proxy.
    exclude_directive_notes` toggle referenced in that same config.yaml
    comment (authority-confusion rationale, not file-relevance) -- that
    toggle is not implemented on this branch (merged separately by another
    lane into a different target than this branch's base). This test pins
    ONLY this lane's own exclusion path: a kind="directive" note anchored to
    the exact window file never reaches the structural channel, regardless
    of whether any other exclusion toggle exists.

    The combined-lanes case this test deliberately does not cover -- both
    mechanisms live at once, and one toggled off while the other stays on --
    is pinned post-merge in tests/test_proactive_p1_composite.py, which
    records the asymmetry the two configs produce together: turning
    `exclude_directive_notes` off does NOT restore directives here, because
    `structural_kinds` removes them upstream of that toggle's own check and
    is not itself channel-scoped."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    directive_nid = svc.remember(
        "always run resolver.py's migration script before deploy", kind="directive",
        anchors=["resolver.py"],
    )
    gotcha_nid = svc.remember(
        "resolver.py holds the workspace lock; drops on scope exit", kind="gotcha",
        anchors=["resolver.py"],
    )
    out = svc.proactive_context(
        text="", file_paths=["/abs/resolver.py"], session_id="s-directive",
    )
    assert f"note:{gotcha_nid}" in out["anchor_ids"]
    assert f"note:{directive_nid}" not in out["anchor_ids"]


def test_structural_overfetch_survives_task_noise_starvation(tmp_path, monkeypatch):
    """Lever 1b: `recall_for_path()`'s OLD fixed `limit == max_items_per_event
    * 2` (6 at the default) would push a single eligible-kind anchored note
    past the truncation cutoff once 6+ more-recent kind="task" notes pile up
    on the same file -- the exact starvation lever 1b exists to fix. Plants
    the gold note FIRST (so it is the oldest), then 6 task notes AFTER (so
    `recall_for_path`'s recency tie-break -- every fresh note ties at
    (author_trust_score, decay_score) == (1.0, 1.0) regardless of kind --
    ranks all 6 ahead of it): under the OLD limit of 6 the gold note falls
    out of the pool entirely (position 7 of 7); under the NEW default
    `min(max_items_per_event * structural_overfetch_multiplier,
    structural_overfetch_ceiling) == min(3*4, 60) == 12` it survives."""
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE="1")
    gold_nid = svc.remember(
        "resolver.py holds the workspace lock; drops on scope exit", kind="gotcha",
        anchors=["resolver.py"],
    )
    for j in range(6):
        svc.remember(f"still investigating resolver.py (session {j})", kind="task")
    out = svc.proactive_context(
        text="", file_paths=["/abs/resolver.py"], session_id="s-overfetch",
    )
    assert f"note:{gold_nid}" in out["anchor_ids"]


# -- recall_scored (UPG-PRO-1) ----------------------------------------------

def test_recall_scored_returns_scores(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    content = "handle_legacy_finalizers appends to gc.garbage when tp_del is set here now"
    svc.remember(content)
    scored = svc.recall_scored(query=content, limit=5)
    assert scored
    note, score = scored[0]
    assert note.content == content
    assert score is not None and 0.0 <= score <= 1.0001


def test_recall_scored_sql_fallback_none(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    svc.remember("some note body")
    scored = svc.recall_scored(query=None, limit=5)  # no query -> SQL path
    assert scored
    assert all(s is None for (_n, s) in scored)  # never a fabricated number


# -- artifact cache (org-wide caching) --------------------------------------

def test_search_cache_hit_and_metrics(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE_CACHE="1")
    assert svc.cache_metrics() is not None  # cache is enabled

    calls = {"n": 0}
    result = SearchResultList([
        SearchResult(file_path="a.py", lines="1-2", symbol_name="foo",
                     language="python", score=0.9, content="def foo(): ...")
    ])

    def _fake_search(query, n_results, language, semantic_weight):
        calls["n"] += 1
        return result, 12

    monkeypatch.setattr(svc._searcher, "search", _fake_search)

    r1, ms1 = svc.search("lock", n_results=5)
    r2, ms2 = svc.search("lock", n_results=5)  # identical -> cache hit
    assert calls["n"] == 1                      # searcher ran once
    assert ms2 == 0                             # hit reports no search time
    assert list(r1) == list(r2)
    m = svc.cache_metrics()
    assert m["hits"] == 1 and m["misses"] == 1


def test_recall_scored_cache_invalidates_on_note_change(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch, VECTR_PROACTIVE_CACHE="1")
    svc.remember("alpha note about the workspace lock resolver flow here")
    q = "alpha note about the workspace lock resolver flow here"
    svc.recall_scored(query=q)          # miss -> stored
    svc.recall_scored(query=q)          # hit
    m1 = svc.cache_metrics()
    assert m1["hits"] == 1
    # A new note bumps the notes epoch -> the prior cache key can never match.
    svc.remember("beta note changes the notes epoch")
    svc.recall_scored(query=q)          # miss again (epoch changed)
    m2 = svc.cache_metrics()
    assert m2["misses"] == m1["misses"] + 1
