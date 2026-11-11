"""
Shared fixtures for the vectr test suite.

Key fixture: `indexer` — real CodeIndexer backed by an in-memory-style ChromaDB
(stored in tmp_path) with the heavy sentence-transformers model replaced by a
deterministic dummy embedder. No model download required; tests run in <1 s.
"""
from __future__ import annotations

import os
import sys
import textwrap
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

# langchain_community 0.4+ removed chat_models.vertexai (moved to langchain-google-vertexai).
# Stub it so ragas can be imported without requiring the VertexAI extras.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

# Disable cross-encoder reranker before any searcher import so tests never
# trigger a model download.
os.environ["VECTR_RERANKER_MODEL"] = ""

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Saved at collection time (before any fixture patches app.service.VectrService).
# real_service_client patches that name session-wide; test_ragas_eval uses this
# reference so it always gets the real constructor, not the mock.
from app.service import VectrService as _RealVectrService


# ---------------------------------------------------------------------------
# UPG-TEST-CACHE-ISOLATION: redirect the whole session off the real user
# cache (~/.cache/vectr).
#
# Every product cache-path resolution — VectrService's workspace DB dir
# (app/service.py:_default_db_dir), CodeIndexer's own db_path fallback
# (agent/indexer/_core.py), and the Hugging Face model cache (agent/indexer/
# _types.py, agent/searcher.py) — goes through agent.config.vectr_cache_root(),
# which reads VECTR_CACHE_DIR at CALL time (never cached at import). Setting
# that env var once here, before the first test runs, is therefore enough to
# isolate every VectrService(workspace_root=str(tmp_path)) / CodeIndexer(...)
# construction anywhere in the suite — hundreds of call sites across dozens
# of test files, present and future — with no change needed to any of them:
# this fixture is autouse, so no test signature has to name it.
#
# Before vectr_cache_root() existed, each of those constructions used a
# per-test-unique tmp_path as its workspace root, hashed to a unique cache
# slug, and wrote a real directory under ~/.cache/vectr for every test that
# ran — a 2026-07-20 cleanup swept ~4,000 such junk dirs (~550 MB) left by
# prior suite runs.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _isolated_cache_root(tmp_path_factory) -> Generator[Path, None, None]:
    from agent.config import CACHE_DIR_ENV

    cache_dir = tmp_path_factory.mktemp("vectr_cache_root")
    # A session-scoped fixture cannot request the function-scoped `monkeypatch`
    # fixture, so this uses pytest's own MonkeyPatch class directly — the
    # documented pattern for env-var patching outside a test function
    # (https://docs.pytest.org/en/stable/how-to/monkeypatch.html).
    mp = pytest.MonkeyPatch()
    mp.setenv(CACHE_DIR_ENV, str(cache_dir))
    try:
        yield cache_dir
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# Dummy embed provider — deterministic, zero-download
# ---------------------------------------------------------------------------

class _DummyEmbedProvider:
    """Deterministic 768-dim embedder for unit tests. Matches nomic-embed-code dim."""
    DIM = 768

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            seed = abs(hash(text[:80])) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.DIM).astype(np.float32)
            norm = np.linalg.norm(v)
            out.append(v / (norm + 1e-8))
        return np.array(out)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts).tolist()

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        # Symmetric stand-in — no registered query prompt, so query-mode embedding
        # is identical to document-mode embedding (matches most real embed models).
        return self.embed(texts)


@pytest.fixture
def indexer(tmp_path, monkeypatch):
    """
    CodeIndexer backed by a fresh ChromaDB in tmp_path.
    The embed provider is replaced with _DummyEmbedProvider — no model download.
    """
    from agent import indexer as idx_module
    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider())
    from agent.indexer import CodeIndexer
    return CodeIndexer(
        workspace_root=str(tmp_path),
        db_path=str(tmp_path / "chroma"),
    )


@pytest.fixture
def searcher(indexer):
    """CodeSearcher wrapping a mocked-embedder CodeIndexer."""
    from agent.searcher import CodeSearcher
    return CodeSearcher(indexer)


# ---------------------------------------------------------------------------
# UPG-CONFTEST-SERVICE-CLOBBER: universal app.state.service snapshot/restore.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_app_state_service():
    """Snapshot and restore ``app.state.service`` around every test so no fixture
    or inline ``TestClient`` block can leave a mock/partial service installed for
    a later test to exercise vacuously (the "lying mock" class).

    The session-scoped ``real_service_client`` sets ``app.state.service`` once and
    relies on it persisting; a mock-based test that ran in between used to clobber
    it and never restore, so a later real-service REST test silently exercised the
    wrong service. Rolling the value back after every test keeps whatever was
    installed at each test's start (the real service, once ``real_service_client``
    is set up) authoritative — which is what makes removing the local
    ``_reaffirm_real_service`` workaround safe. Higher-scoped fixtures set up
    before this function-scoped autouse, so the snapshot already reflects them."""
    from api import app
    saved = getattr(app.state, "service", None)
    try:
        yield
    finally:
        app.state.service = saved


# ---------------------------------------------------------------------------
# Real-service fixture — full pipeline with dummy embedder
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_service_client(tmp_path_factory):
    """
    FastAPI TestClient backed by a REAL VectrService with dummy embedder.

    Unlike `client` (which mocks the entire service), this exercises the full
    pipeline: HTTP → routes → VectrService → CodeIndexer → ChromaDB →
    CodeSearcher (BM25 + vector) → memory store.

    The embed provider is the deterministic dummy so no model download is
    needed, but everything else is production code.

    Important: the lifespan handler in api.py creates its own VectrService on
    TestClient entry and sets app.state.service.  We prevent that from clobbering
    our pre-built svc by patching VectrService in app.service so the lifespan
    call returns *our* svc instead of creating a fresh one pointed at the repo.
    """
    tmp = tmp_path_factory.mktemp("real_svc")

    # Both env vars are read only once, synchronously, inside VectrService.__init__
    # (app/service.py: VECTR_EMBED_MODEL at self._embed_model assignment,
    # VECTR_DB_DIR at db_dir resolution) — never re-read afterward. Scoping this
    # patch to just the constructor call (like the app.service.VectrService patch
    # below already does) prevents it from leaking VECTR_EMBED_MODEL=dummy into
    # os.environ for the rest of the test session: this fixture is session-scoped,
    # so a `with` block wrapped around the `yield` would only restore os.environ at
    # session teardown, after every other test file has already run with it set.
    with patch("agent.indexer.get_embed_provider", return_value=_DummyEmbedProvider()):
        from app.service import VectrService
        from api import app

        with patch.dict("os.environ", {"VECTR_DB_DIR": str(tmp), "VECTR_EMBED_MODEL": "dummy"}):
            svc = VectrService(workspace_root=str(tmp))

        # Patch `app.service.VectrService` only across TestClient startup, where the
        # `lifespan` handler's own `VectrService(...)` call must be intercepted to
        # return our pre-built `svc` instead of constructing a fresh one pointed at
        # the real repo. Scoping the patch to just __enter__() (rather than wrapping
        # it around the whole `with` block, which — for a session-scoped generator
        # fixture — would keep the patch active for the REST OF THE TEST SESSION)
        # prevents every later test's unrelated `VectrService(...)` construction
        # from silently being redirected to this one shared, ever-growing instance.
        c = TestClient(app, raise_server_exceptions=True)
        with patch("app.service.VectrService", return_value=svc):
            c.__enter__()
        try:
            yield c, svc, str(tmp)
        finally:
            c.__exit__(None, None, None)
            svc.shutdown()  # release the indexer's ChromaDB client at session end


# ---------------------------------------------------------------------------
# Integration fixture — real nomic-embed-code model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def integration_indexer(tmp_path_factory):
    """
    CodeIndexer with the production Snowflake/snowflake-arctic-embed-m-v1.5 model.

    Downloads once (~440 MB), then cached at ~/.cache/vectr/models.
    Used only by @pytest.mark.integration tests.  Run with: pytest -m integration
    """
    import os as _os
    tmp = tmp_path_factory.mktemp("integration")
    model = _os.getenv("VECTR_EMBED_MODEL", "Snowflake/snowflake-arctic-embed-m-v1.5")
    from agent.indexer import CodeIndexer
    return CodeIndexer(
        workspace_root=str(tmp),
        embed_model=model,
        db_path=str(tmp / "chroma"),
    )


# ---------------------------------------------------------------------------
# Python file factory helper
# ---------------------------------------------------------------------------

def make_py(tmp_path: Path, name: str, content: str) -> str:
    """Write a Python file into tmp_path and return its absolute path."""
    f = tmp_path / name
    f.write_text(textwrap.dedent(content))
    return str(f)


# ---------------------------------------------------------------------------
# FastAPI client fixtures
# ---------------------------------------------------------------------------

def _base_mock_service():
    """Mock VectrService with sensible defaults for API route tests."""
    from agent.searcher import SearchResult
    from agent.symbol_graph import LocateResult

    svc = MagicMock()
    svc._embed_model = "BAAI/bge-base-en-v1.5"
    svc.total_chunks = 100
    # UPG-8.2: /v1/health sources last_indexed from the same VectrService
    # property that populates svc.status()["last_indexed"] below.
    svc.last_indexed = "2026-01-01T00:00:00Z"

    _result = SearchResult(
        file_path="src/auth.py", lines="10-30", symbol_name="verify_token",
        language="python", score=0.91, content="def verify_token(): ...",
    )
    svc.search.return_value = ([_result], 15)
    # UPG-QUERYTYPE-REROUTE: additive symbol-graph hint — no exact identifier
    # match by default, so no hint section is appended in the common case.
    svc.identifier_hint_symbols.return_value = []
    svc.index.return_value = (5, 100, 120)
    svc.status.return_value = {
        "indexed_files": 5, "total_chunks": 100,
        "last_indexed": "2026-01-01T00:00:00Z",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "workspace_root": "/repo", "symbol_count": 20,
    }
    svc.get_map.return_value = "# Passport\nPython FastAPI service."
    # Real locate_with_snippets() returns a LocateResult, not a bare list —
    # a mock returning [] made /v1/locate 500 on `result.symbols` for any
    # test that actually asserted a 200 (caught while adding search-only
    # REST coverage; conftest mock was the lone type mismatch here).
    svc.locate_with_snippets.return_value = LocateResult(symbols=[], resolution_strategy="none", query="")
    svc.format_locate.return_value = "No results."
    svc.trace_with_snippets.return_value = {}
    svc.format_trace.return_value = "No trace."
    svc.should_evict.return_value = False
    svc.eviction_hint.return_value = ""
    svc.remember.return_value = 1
    # /v1/remember dispatches remember_with_extras(), not remember() — a bare
    # MagicMock return here would validate against RememberResponse only by
    # accident (MagicMock's default __int__/__iter__ stubs), while an f-string
    # embedding outcome.note_id would still leak a "<MagicMock ...>" repr into
    # the confirmation message. Return the REAL outcome type.
    from app.service import RememberOutcome
    svc.remember_with_extras.return_value = RememberOutcome(
        note_id=1, related=[], revoked_related=[], proxy_anchor_suggestions=[],
    )
    svc.promote_note.return_value = True
    svc.revoke_note.return_value = True
    svc.reinstate_note.return_value = True
    svc.recall.return_value = "# Working Notes (1 entries)\n\n[1] [HIGH] test content\n"
    svc.snapshot_session.return_value = "snap_abc123"
    svc.list_snapshots.return_value = [{"snapshot_id": "snap_abc123", "label": "test", "created_at": 0.0}]
    # UPG-RESUME-SURFACE: real VectrService.resume() shape (last_task/gotchas/
    # snapshot/formatted) — a bare MagicMock would fail `ResumeResponse(**data)`
    # at the route (mocks must return the REAL type, not a stand-in).
    svc.resume.return_value = {
        "last_task": None,
        "gotchas": [],
        "snapshot": None,
        "formatted": (
            "Nothing to resume yet — no task notes, snapshots, or gotchas "
            "recorded for this workspace. Use vectr_remember(kind='task', ...) "
            "to start one."
        ),
    }
    # Default mode is full (not memory-only / not search-only); must be an
    # explicit bool, not a MagicMock (bare MagicMock attrs are truthy by default).
    svc.memory_only = False
    svc.search_only = False
    return svc


@pytest.fixture
def client():
    """FastAPI TestClient with fully-mocked VectrService. Fast — no model loading."""
    from api import app
    svc = _base_mock_service()
    # UPG-CONFTEST-SERVICE-CLOBBER: save/restore app.state.service so this mock
    # never persists past the fixture and clobber a later real-service test that
    # runs after it (a REST test could otherwise exercise a mock and pass
    # vacuously depending on execution order).
    _prior_service = getattr(app.state, "service", None)
    try:
        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                yield c
    finally:
        app.state.service = _prior_service


@pytest.fixture
def client_real_memory(tmp_path):
    """
    FastAPI TestClient where search is mocked but WorkingContextStore is REAL.
    Used to test the full remember → recall round-trip through HTTP without
    loading the embedding model.
    """
    from api import app
    from agent.working_context_store import WorkingContextStore
    from agent.trigger_engine import TriggerFireLedger, TurnInjectionLedger

    svc = _base_mock_service()
    real_store = WorkingContextStore(str(tmp_path))
    ws = str(tmp_path)

    def _remember(content, tags=None, priority="medium", session_id=None, kind="finding", title="",
                  agent="", triggers=None, provenance="agent", scope=None, anchors=None,
                  supersedes=None, contradicts=None, user_quote=None, pin=False):
        return real_store.remember(
            ws, content, tags, priority, session_id, kind=kind, title=title, author_id=agent,
            triggers=triggers, provenance=provenance, scope=scope, anchors=anchors,
            supersedes=supersedes, contradicts=contradicts, user_quote=user_quote,
            pin=pin,
        )

    def _remember_with_extras(content, tags=None, priority="medium", session_id=None,
                               kind="finding", title="", agent="", triggers=None,
                               provenance="agent", scope=None, anchors=None,
                               supersedes=None, contradicts=None, user_quote=None, pin=False):
        """Mirrors VectrService.remember_with_extras's own gating (app/
        service.py) against this fixture's real store, so /v1/remember
        REST tests routed through this fixture get a REAL RememberOutcome
        (real RelatedNote list, real proxy-anchor list) rather than a bare
        MagicMock return. This store has no embedder attached, so `related`
        is always [] here (the same real fail-open path a genuinely
        embedder-less store takes in production); `proxy_anchor_suggestions`
        is real glob presence against `tmp_path`, needing no embedder."""
        from app.service import RememberOutcome
        from agent.proxy_anchors import suggest_proxy_anchors
        from agent.config import (
            MEMORY_WRITE_RELATED_ENABLED,
            MEMORY_WRITE_RELATED_LIMIT,
            MEMORY_WRITE_RELATED_MIN_SIMILARITY,
            MEMORY_WRITE_RELATED_REVOKED_ENABLED,
            MEMORY_WRITE_RELATED_REVOKED_LIMIT,
            MEMORY_WRITE_PROXY_SUGGEST_ENABLED,
            MEMORY_WRITE_PROXY_SUGGEST_LIMIT,
        )
        note_id = _remember(
            content, tags, priority, session_id, kind, title, agent, triggers,
            provenance, scope, anchors, supersedes, contradicts, user_quote,
            pin=pin,
        )
        related = []
        if MEMORY_WRITE_RELATED_ENABLED:
            related = real_store.related_active_notes(
                ws, note_id, limit=MEMORY_WRITE_RELATED_LIMIT,
                min_similarity=MEMORY_WRITE_RELATED_MIN_SIMILARITY,
            )
        revoked_related = []
        if MEMORY_WRITE_RELATED_REVOKED_ENABLED:
            revoked_related = real_store.revoked_related_notes(
                ws, note_id, limit=MEMORY_WRITE_RELATED_REVOKED_LIMIT,
                min_similarity=MEMORY_WRITE_RELATED_MIN_SIMILARITY,
            )
        proxy_anchor_suggestions = []
        if MEMORY_WRITE_PROXY_SUGGEST_ENABLED and kind == "operational" and not anchors:
            proxy_anchor_suggestions = suggest_proxy_anchors(ws, MEMORY_WRITE_PROXY_SUGGEST_LIMIT)
        return RememberOutcome(
            note_id=note_id, related=related, revoked_related=revoked_related,
            proxy_anchor_suggestions=proxy_anchor_suggestions,
        )

    svc.remember.side_effect = _remember
    svc.remember_with_extras.side_effect = _remember_with_extras
    svc.promote_note.side_effect = lambda note_id, to: real_store.promote(ws, note_id, to)
    svc.revoke_note.side_effect = lambda note_id, reason, actor="agent": real_store.revoke_note(
        ws, note_id, reason, actor=actor
    )
    svc.reinstate_note.side_effect = lambda note_id, actor="agent", reason=None: real_store.reinstate_note(
        ws, note_id, actor=actor, reason=reason
    )
    # 1.12.0 routes: a MagicMock here answers 200 for a note that does not
    # exist, so wire them to the real store like the other note mutations.
    svc.pin_note.side_effect = lambda note_id, pinned=True: real_store.set_pinned(
        ws, note_id, pinned
    )
    svc.supersede_note.side_effect = lambda note_id, superseded_by=None, reason=None, actor="agent": real_store.supersede_note(
        ws, note_id, superseded_by=superseded_by, reason=reason, actor=actor
    )

    # TRIGGER-ENGINE wave 2a: a minimal per-session ledger registry mirroring
    # `VectrService._ledger_for`/`reset_trigger_ledger` so REST-level tests
    # against this REAL store can exercise fire-dedup and cumulative budget
    # through the actual `/v1/recall` request/response cycle, not just a
    # stand-in that silently accepts and drops `session_id`/`events`.
    _ledgers: dict[str, TriggerFireLedger] = {}

    def _ledger_for(session_id):
        if not session_id:
            return None
        return _ledgers.setdefault(session_id, TriggerFireLedger())

    def _reset_trigger_ledger(session_id):
        if session_id and session_id in _ledgers:
            _ledgers[session_id].reset()

    svc.reset_trigger_ledger.side_effect = _reset_trigger_ledger

    # Serving-policy hardening (wave 3, §5.3/§5.4): a second, TURN-scoped
    # registry mirroring `VectrService._turn_ledger_for`/`reset_turn_ledger`
    # alongside `_ledgers`/`_ledger_for` above. Without this, a REST test
    # against this REAL store could never exercise cross-surface same-turn
    # dedup or the shared ≤500-token turn budget (`fire_and_format`'s
    # `turn_ledger=None` default silently reproduces the pre-wave-3
    # unbounded/undeduped behaviour, the same way `ledger=None` reproduces
    # pre-wave-2a behaviour) — every `/v1/recall` request would look like
    # its own fresh turn even when two hook surfaces fire within one.
    _turn_ledgers: dict[str, TurnInjectionLedger] = {}

    def _turn_ledger_for(session_id):
        if not session_id:
            return None
        return _turn_ledgers.setdefault(session_id, TurnInjectionLedger())

    def _reset_turn_ledger(session_id):
        if session_id and session_id in _turn_ledgers:
            _turn_ledgers[session_id].reset()

    svc.reset_turn_ledger.side_effect = _reset_turn_ledger

    def _recall(query=None, tags=None, priority=None, limit=10, kind=None, boot=False,
                min_similarity=None, file_path=None, command=None, max_age_days=None, sort_by="relevance",
                detail="index", note_id=None, surface="mcp", hook_event=None,
                session_id=None, events=None):
        if note_id is not None:
            note = real_store.get_note(ws, note_id)
            if note is None:
                return f"Note #{note_id} not found."
            stale = real_store.check_staleness([note], ws)
            return real_store.format_notes_for_llm([note], stale_warnings=stale, detail="full", surface=surface)
        if boot:
            events_to_fire = events if events else ["session-start"]
            # `spend_turn_budget` deliberately omitted (defaults False),
            # mirroring `VectrService._recall_impl`'s boot branch exactly:
            # session-start bulk keeps its own separate per-SESSION cap
            # (`ledger`) rather than the smaller ordinary-turn allowance —
            # the turn ledger's dedup CLAIM still runs via `turn_ledger`,
            # so a note delivered at boot is still excluded from a same-
            # turn PreToolUse/prompt-submit re-delivery.
            fire_text, _ = real_store.fire_and_format(
                ws, events=events_to_fire, session_id=session_id,
                ledger=_ledger_for(session_id), turn_ledger=_turn_ledger_for(session_id),
                surface=surface,
            )
            return fire_text
        if file_path:
            turn_ledger = _turn_ledger_for(session_id)
            fire_text, fired_ids = real_store.fire_and_format(
                ws, event="pre-edit", file_path=file_path, session_id=session_id,
                ledger=_ledger_for(session_id), turn_ledger=turn_ledger,
                spend_turn_budget=True, surface=surface,
            )
            path_notes = real_store.recall_for_path(ws, file_path, kind=kind, limit=limit, session_id=session_id)
            # Mirrors VectrService._recall_impl's own fix: a note claimed by
            # an EARLIER surface this turn (turn-deduped out of `fired_ids`
            # here) must also be excluded from the legacy content-match
            # fallback, not just from this call's own engine delivery.
            path_notes = [
                n for n in path_notes
                if n.note_id not in fired_ids
                and (turn_ledger is None or turn_ledger.eligible(n.note_id))
            ]
            legacy_text = real_store.format_notes_for_llm(path_notes, detail=detail, surface=surface) if path_notes else ""
            if fire_text and legacy_text:
                return fire_text + "\n\n" + legacy_text
            return fire_text or legacy_text
        if command:
            fire_text, _ = real_store.fire_and_format(
                ws, event="pre-run", command=command, session_id=session_id,
                ledger=_ledger_for(session_id), turn_ledger=_turn_ledger_for(session_id),
                spend_turn_budget=True, surface=surface,
            )
            return fire_text
        fire_text, fired_ids = "", set()
        turn_ledger = _turn_ledger_for(session_id)
        if events:
            fire_text, fired_ids = real_store.fire_and_format(
                ws, events=events, session_id=session_id,
                ledger=_ledger_for(session_id), turn_ledger=turn_ledger,
                spend_turn_budget=True, surface=surface,
            )
        notes = real_store.recall(ws, query, tags, priority, limit, kind=kind, min_similarity=min_similarity,
                                  max_age_days=max_age_days, sort_by=sort_by, session_id=session_id)
        # Mirrors VectrService._recall_impl's own fix: exclude notes already
        # claimed this turn by an EARLIER surface, not just this call's own
        # engine delivery — but ONLY when `events` is given, i.e. this call
        # itself stands in for an injection surface. A plain direct
        # `vectr_recall(query=...)` (no `events`) must never be turn-deduped
        # — it is a deliberate lookup, not a passive injection surface, and
        # `fired_ids` is always empty here when `events` is falsy anyway.
        if events:
            notes = [
                n for n in notes
                if n.note_id not in fired_ids
                and (turn_ledger is None or turn_ledger.eligible(n.note_id))
            ]
        formatted = real_store.format_notes_for_llm(notes, detail=detail, surface=surface, sort_by=sort_by)
        if fire_text and formatted:
            return fire_text + "\n\n" + formatted
        return fire_text or formatted

    svc.recall.side_effect = _recall
    svc.forget_note.side_effect = lambda note_id: real_store.forget(ws, note_id)
    svc.forget_all.side_effect = lambda: real_store.forget_all(ws)
    svc.snapshot_session.side_effect = lambda label, session_id=None: \
        real_store.snapshot(ws, label=label)
    svc.list_snapshots.side_effect = lambda: real_store.list_snapshots(ws)

    # UPG-CONFTEST-REAL-MEMORY-MIRRORS: resume() and record_commit_note() are
    # bound straight to the REAL VectrService methods (app/service.py) rather
    # than hand-duplicated here, so a bug in either real method fails these
    # tests instead of silently passing against a parallel reimplementation
    # that could drift from it. The real methods only touch
    # self._context_store / self._workspace_root / self._search_only (plus,
    # for record_commit_note, self._current_task_note()/self._require_memory_
    # layer()/self.remember() — all bound the same way below), so those are
    # set as real attributes on the mock `svc` and the real unbound methods
    # are called against it (svc.remember already delegates to the real
    # store via `_remember` above, so record_commit_note's internal
    # `self.remember(...)` call composes with it for free).
    svc._context_store = real_store
    svc._workspace_root = ws
    svc._search_only = False
    svc._require_memory_layer = lambda: _RealVectrService._require_memory_layer(svc)
    svc._current_task_note = lambda: _RealVectrService._current_task_note(svc)
    svc.resume.side_effect = lambda session_id=None, surface="mcp": _RealVectrService.resume(
        svc, session_id=session_id, surface=surface
    )
    svc.record_commit_note.side_effect = lambda sha, subject, branch, files: (
        _RealVectrService.record_commit_note(svc, sha, subject, branch, files)
    )

    # UPG-CONFTEST-SERVICE-CLOBBER: save/restore app.state.service (see the
    # `client` fixture) so this partial-real service does not persist into a
    # later test that relies on a different app.state.service.
    _prior_service = getattr(app.state, "service", None)
    try:
        with patch("app.service.VectrService", return_value=svc):
            with TestClient(app, raise_server_exceptions=True) as c:
                app.state.service = svc
                yield c
    finally:
        app.state.service = _prior_service
