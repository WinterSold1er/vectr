"""Tests for agent/chroma_dispatch.py (UPG-CHROMA-BLOCKING-EVENT-LOOP,
UPG-NOTES-CHROMA-BLOCKING-EVENT-LOOP).

Every REST route and every MCP tool dispatch is served by the same request
event loop, so a synchronous call into the vector store (query/get/upsert/
delete) can hold that loop for as long as the store's own internal work
takes. `dispatch_chroma_sync`/`dispatch_chroma_async` route such a call
through a dedicated single-worker executor instead of running it in place.

Covers:
- Unit-level: both dispatch functions actually run `fn` off the calling
  thread, against both a real single-worker executor and a service double
  with none attached (`dispatch_chroma_async` must never fall back to an
  in-place call; `dispatch_chroma_sync`'s in-place fallback is safe because
  its callers are already off-loop before it is invoked).
- Exception propagation: a dispatched call that raises must surface the
  ORIGINAL exception type to the caller, not a wrapper.
- End-to-end regression for the reported outage: a slow, real ChromaDB
  `collection.count()` call must never delay a concurrent `GET /v1/status`,
  using a REAL VectrService and REAL collection (not a mocked service
  method) so the slow call is the actual chroma operation named in the bug,
  not a stand-in.
- Working-memory (notes-collection) request paths — every REST route and
  MCP tool that reaches `WorkingContextStore._notes_col` — actually run the
  underlying `VectrService` method off the calling thread, on the service's
  own dedicated `_chroma_executor` (the SAME single-worker executor the
  code-collection paths above already share; a second executor would newly
  allow concurrent notes+search chroma access, the exact race the shared
  single worker exists to prevent).
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.chroma_dispatch import dispatch_chroma_async, dispatch_chroma_sync
from tests.conftest import _DummyEmbedProvider, make_py


# ---------------------------------------------------------------------------
# Unit-level dispatch behavior
# ---------------------------------------------------------------------------

class TestDispatchChromaSync:
    def test_runs_on_the_services_own_executor_thread_not_the_caller_thread(self) -> None:
        service = MagicMock()
        service._chroma_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-chroma-sync")
        caller_thread = threading.current_thread()
        try:
            seen_thread = dispatch_chroma_sync(service, threading.current_thread)
            assert seen_thread is not caller_thread
            assert seen_thread.name.startswith("test-chroma-sync")
        finally:
            service._chroma_executor.shutdown(wait=True)

    def test_falls_back_to_an_in_place_call_without_a_real_executor(self) -> None:
        # A service double with no dedicated executor attached (a MagicMock
        # auto-vivifies `_chroma_executor` as another MagicMock, not a real
        # ThreadPoolExecutor) is safe to call in place: every
        # dispatch_chroma_sync caller (the MCP transport's synchronous tool
        # dispatcher) is already off the event loop by the time this
        # function is invoked.
        service = MagicMock()
        service._chroma_executor = MagicMock()
        assert dispatch_chroma_sync(service, lambda x: x * 2, 21) == 42


class TestDispatchChromaAsync:
    async def test_runs_on_the_services_own_executor_thread_not_the_event_loop_thread(self) -> None:
        service = MagicMock()
        service._chroma_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-chroma-async")
        loop_thread = threading.current_thread()
        try:
            seen_thread = await dispatch_chroma_async(service, threading.current_thread)
            assert seen_thread is not loop_thread
            assert seen_thread.name.startswith("test-chroma-async")
        finally:
            service._chroma_executor.shutdown(wait=True)

    async def test_never_calls_fn_in_place_without_a_real_executor(self) -> None:
        # UPG-CHROMA-BLOCKING-EVENT-LOOP regression: unlike the sync variant,
        # the async variant must still dispatch off the calling coroutine's
        # event-loop thread even for a service double with no real executor
        # attached — an in-place fallback here would block the loop for
        # exactly the mocked-service scenario this bug was first caught in
        # (test_index_does_not_block_concurrent_status in test_api.py).
        service = MagicMock()
        service._chroma_executor = MagicMock()
        loop_thread = threading.current_thread()
        seen_thread = await dispatch_chroma_async(service, threading.current_thread)
        assert seen_thread is not loop_thread


# ---------------------------------------------------------------------------
# End-to-end regression — real service, real ChromaDB, real slow call
# ---------------------------------------------------------------------------

def _make_real_service(tmp_path, monkeypatch):
    """Mirrors `_make_real_service` in tests/test_arc_distill.py — a real
    VectrService, dummy embedder, own workspace root."""
    from agent import indexer as idx_module

    monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
    with patch("integrations.vscode_bridge.configure_all"), \
         patch("integrations.workspace_detect.find_workspace_root", return_value=str(tmp_path)), \
         patch.dict("os.environ", {
             "VECTR_DB_DIR": str(tmp_path / "db"),
             "VECTR_EMBED_MODEL": "dummy",
         }):
        from app.service import VectrService
        return VectrService(workspace_root=str(tmp_path))


class TestStatusSurvivesASlowRealChromaCall:
    def test_status_stays_fast_while_a_real_collection_count_call_is_blocked(
        self, tmp_path, monkeypatch
    ) -> None:
        """Reproduces the reported outage: a live daemon froze for 41 minutes
        because GET /v1/status called self._collection.count() live on every
        read while chroma's own internal work (compacting a large
        collection) was blocking the event loop. status() must never touch
        chroma at request time — total_chunks is a cache refreshed only at
        the end of a mutation method (index_file/delete_file/
        index_workspace), so a slow/blocked collection.count() call
        triggered by one in-flight mutation must never delay a concurrent
        status read."""
        make_py(tmp_path, "a.py", "def a(): pass")
        svc = _make_real_service(tmp_path, monkeypatch)
        try:
            # First index establishes the embed-model stamp on the collection
            # (UPG-EMBEDDER-SWAP-GRANITE) — a bare-new db dir's first-ever
            # index call always recreates the collection object regardless of
            # config, which would silently discard a .count() patch applied
            # beforehand. Patching after this call targets the stable
            # collection object subsequent calls actually reuse.
            svc.index(str(tmp_path))

            real_count = svc._indexer._collection.count
            entered = threading.Event()
            release = threading.Event()

            def slow_count():
                entered.set()
                assert release.wait(timeout=5), "test never released the slow count() call"
                return real_count()

            make_py(tmp_path, "b.py", "def b(): pass")
            with patch.object(svc._indexer._collection, "count", side_effect=slow_count):
                from api import app
                prior = getattr(app.state, "service", None)
                with patch("app.service.VectrService", return_value=svc):
                    with TestClient(app, raise_server_exceptions=True) as client:
                        app.state.service = svc
                        try:
                            results: dict[str, object] = {}

                            def do_index() -> None:
                                results["resp"] = client.post(
                                    "/v1/index", json={"path": str(tmp_path), "force": False},
                                )

                            t = threading.Thread(target=do_index)
                            t.start()
                            assert entered.wait(timeout=2), (
                                "slow collection.count() never entered — "
                                "index_workspace's end-of-call refresh may not "
                                "have run"
                            )

                            t0 = time.monotonic()
                            status_resp = client.get("/v1/status")
                            elapsed = time.monotonic() - t0

                            release.set()
                            t.join(timeout=5)
                        finally:
                            app.state.service = prior

            assert status_resp.status_code == 200
            assert elapsed < 1.0, (
                f"/v1/status took {elapsed:.2f}s while collection.count() was "
                "blocked — status must never call chroma at request time"
            )
            assert status_resp.json()["total_chunks"] >= 0
            assert results["resp"].status_code == 200
        finally:
            svc.shutdown()


# ---------------------------------------------------------------------------
# Exception propagation through dispatch
# ---------------------------------------------------------------------------

class TestDispatchExceptionPropagation:
    """A dispatched call that raises must surface the ORIGINAL exception type
    to the caller — `Future.result()`/`run_in_executor` already re-raise
    verbatim, but this pins that contract explicitly per the requirement
    that a route's own `except ValueError` (etc.) handling keeps working
    once the call underneath it is dispatched off-loop."""

    def test_dispatch_chroma_sync_reraises_the_original_exception_type(self) -> None:
        service = MagicMock()
        service._chroma_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-sync-exc")

        def _boom():
            raise ValueError("bad trigger spec")

        try:
            with pytest.raises(ValueError, match="bad trigger spec"):
                dispatch_chroma_sync(service, _boom)
        finally:
            service._chroma_executor.shutdown(wait=True)

    async def test_dispatch_chroma_async_reraises_the_original_exception_type(self) -> None:
        service = MagicMock()
        service._chroma_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-async-exc")

        def _boom():
            raise ValueError("bad trigger spec")

        try:
            with pytest.raises(ValueError, match="bad trigger spec"):
                await dispatch_chroma_async(service, _boom)
        finally:
            service._chroma_executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Working-memory (notes-collection) request paths dispatch off-loop
# ---------------------------------------------------------------------------

def _memory_mock_service() -> MagicMock:
    """A MagicMock VectrService with a REAL single-worker executor attached
    (mirrors the shared `_chroma_executor` every real `VectrService` owns)
    so a route/tool handler that dispatches through it can be told apart
    from one that calls the underlying method in place."""
    svc = MagicMock()
    svc._chroma_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-notes-chroma")
    svc.search_only = False
    svc.memory_only = False
    return svc


def _capture_thread(return_value):
    """Build a side_effect that records the calling thread and returns
    `return_value`, for asserting a mocked service method actually ran on
    the dispatch executor's worker thread rather than the caller's own."""
    seen: dict[str, threading.Thread] = {}

    def _side_effect(*args, **kwargs):
        seen["thread"] = threading.current_thread()
        return return_value

    return seen, _side_effect


class TestMemoryRoutesDispatchOffLoop:
    """Every REST memory route that reaches the notes chroma collection must
    invoke the underlying VectrService method on the dedicated
    `_chroma_executor` worker thread, never on the request's own event-loop
    thread — the same off-loop guarantee UPG-CHROMA-BLOCKING-EVENT-LOOP
    already gave `/v1/search` and `/v1/fetch`."""

    def _run(self, svc: MagicMock, request_fn):
        from api import app

        prior = getattr(app.state, "service", None)
        try:
            with patch("app.service.VectrService", return_value=svc):
                with TestClient(app, raise_server_exceptions=True) as client:
                    app.state.service = svc
                    return request_fn(client)
        finally:
            app.state.service = prior
            svc._chroma_executor.shutdown(wait=True)

    def test_remember_dispatches_off_loop(self) -> None:
        from app.service import RememberOutcome
        svc = _memory_mock_service()
        seen, effect = _capture_thread(
            RememberOutcome(note_id=42, related=[], revoked_related=[], proxy_anchor_suggestions=[]),
        )
        svc.remember_with_extras.side_effect = effect
        resp = self._run(svc, lambda c: c.post("/v1/remember", json={"content": "a note"}))
        assert resp.status_code == 200
        assert resp.json()["note_id"] == 42
        assert seen["thread"].name.startswith("test-notes-chroma")
        assert seen["thread"] is not threading.main_thread()

    def test_recall_dispatches_off_loop(self) -> None:
        svc = _memory_mock_service()
        seen, effect = _capture_thread("# Working Notes\n")
        svc.recall.side_effect = effect
        resp = self._run(svc, lambda c: c.post("/v1/recall", json={"query": "auth"}))
        assert resp.status_code == 200
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_forget_by_note_id_dispatches_off_loop(self) -> None:
        svc = _memory_mock_service()
        seen, effect = _capture_thread(True)
        svc.forget_note.side_effect = effect
        resp = self._run(svc, lambda c: c.post("/v1/forget", json={"note_id": 7}))
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_forget_all_dispatches_off_loop(self) -> None:
        svc = _memory_mock_service()
        seen, effect = _capture_thread(3)
        svc.forget_all.side_effect = effect
        resp = self._run(svc, lambda c: c.post("/v1/forget", json={"all": True}))
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 3
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_memory_clear_dispatches_off_loop(self) -> None:
        svc = _memory_mock_service()
        seen, effect = _capture_thread(5)
        svc.forget_all.side_effect = effect
        resp = self._run(svc, lambda c: c.post("/v1/memory/clear"))
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 5
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_commit_note_dispatches_off_loop(self) -> None:
        svc = _memory_mock_service()
        seen, effect = _capture_thread(9)
        svc.record_commit_note.side_effect = effect
        resp = self._run(
            svc,
            lambda c: c.post(
                "/v1/commit-note",
                json={"sha": "abc1234", "subject": "fix bug", "branch": "main", "files": ["a.py"]},
            ),
        )
        assert resp.status_code == 200
        assert resp.json()["note_id"] == 9
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_proactive_dispatches_off_loop(self) -> None:
        svc = _memory_mock_service()
        seen, effect = _capture_thread(
            {"context": "some context", "item_count": 1, "anchor_ids": ["#1"], "scores": [0.9]}
        )
        svc.proactive_context.side_effect = effect
        resp = self._run(
            svc,
            lambda c: c.post("/v1/proactive", json={"text": "working on auth.py"}),
        )
        assert resp.status_code == 200
        assert resp.json()["context"] == "some context"
        assert seen["thread"].name.startswith("test-notes-chroma")


class TestMemoryMcpToolsDispatchOffLoop:
    """The MCP memory tools (`vectr_remember`, `vectr_recall`, `vectr_forget`)
    are already reached off the event loop via the transport's own
    threadpool hop (`run_in_threadpool` around `handle_tools_call` —
    app/routes.py), so they were never at risk of freezing the request event
    loop the way the un-dispatched REST routes above were. They still belong
    on the SAME single-worker `_chroma_executor` as every other chroma call
    for the same request-at-a-time serialization guarantee MCP
    vectr_search/vectr_fetch already have (UPG-CHROMA-BLOCKING-EVENT-LOOP)."""

    def test_vectr_remember_dispatches_through_the_shared_executor(self) -> None:
        from integrations.mcp_server._dispatch import handle_tools_call
        from app.service import RememberOutcome

        svc = _memory_mock_service()
        seen, effect = _capture_thread(
            RememberOutcome(note_id=11, related=[], revoked_related=[], proxy_anchor_suggestions=[]),
        )
        svc.remember_with_extras.side_effect = effect
        svc.get_note.return_value = None
        try:
            result = handle_tools_call(
                "vectr_remember", {"content": "a note"}, svc, session_id="s1",
            )
        finally:
            svc._chroma_executor.shutdown(wait=True)
        assert result["isError"] is False
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_vectr_recall_dispatches_through_the_shared_executor(self) -> None:
        from integrations.mcp_server._dispatch import handle_tools_call

        svc = _memory_mock_service()
        seen, effect = _capture_thread("# Working Notes\n")
        svc.recall.side_effect = effect
        svc.auto_eviction_hint.return_value = ""
        try:
            result = handle_tools_call(
                "vectr_recall", {"query": "auth"}, svc, session_id="s1",
            )
        finally:
            svc._chroma_executor.shutdown(wait=True)
        assert result["isError"] is False
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_vectr_forget_by_note_id_dispatches_through_the_shared_executor(self) -> None:
        from integrations.mcp_server._dispatch import handle_tools_call

        svc = _memory_mock_service()
        seen, effect = _capture_thread(True)
        svc.forget_note.side_effect = effect
        try:
            result = handle_tools_call(
                "vectr_forget", {"note_id": 7}, svc, session_id="s1",
            )
        finally:
            svc._chroma_executor.shutdown(wait=True)
        assert result["isError"] is False
        assert seen["thread"].name.startswith("test-notes-chroma")

    def test_vectr_forget_all_dispatches_through_the_shared_executor(self) -> None:
        from integrations.mcp_server._dispatch import handle_tools_call

        svc = _memory_mock_service()
        seen, effect = _capture_thread(4)
        svc.forget_all.side_effect = effect
        try:
            result = handle_tools_call(
                "vectr_forget", {"all": True}, svc, session_id="s1",
            )
        finally:
            svc._chroma_executor.shutdown(wait=True)
        assert result["isError"] is False
        assert seen["thread"].name.startswith("test-notes-chroma")
