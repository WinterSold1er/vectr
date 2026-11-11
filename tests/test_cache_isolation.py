"""UPG-TEST-CACHE-ISOLATION: the pytest suite must never write to the real
user cache (~/.cache/vectr).

Every product cache-path resolution funnels through the single override
point `agent.config.vectr_cache_root()` (VECTR_CACHE_DIR env var, read at
call time). `tests/conftest.py`'s session-scoped autouse `_isolated_cache_root`
fixture sets that env var to a tmp_path before the first test runs, which is
enough to isolate every VectrService / CodeIndexer construction in the suite
with no per-test changes.

This file is the CI assertion that the isolation actually holds. It does NOT
diff the real ~/.cache/vectr directory listing before/after the run: this
suite frequently runs alongside other vectr worktrees/instances that
legitimately write to the real cache concurrently (e.g. the always-on daemon,
sibling test runs on the shared machine), so a raw directory-listing diff
would be racy. Instead these tests assert, deterministically, that every
call site resolves under the *isolated* root and never under the real one --
true regardless of what else is concurrently touching the real cache.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _real_home_cache_root() -> Path:
    """The real ~/.cache/vectr location, computed the same way
    agent.config.vectr_cache_root() computes its un-overridden fallback --
    independent of whatever VECTR_CACHE_DIR the session fixture has set."""
    return Path(os.path.expanduser("~")) / ".cache" / "vectr"


class TestVectrCacheRootResolver:
    """agent.config.vectr_cache_root(): the single override point every
    cache-path callsite routes through."""

    def test_honors_env_override(self, tmp_path, monkeypatch) -> None:
        from agent.config import CACHE_DIR_ENV, vectr_cache_root

        monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
        assert vectr_cache_root() == tmp_path

    def test_falls_back_to_home_when_unset(self, tmp_path, monkeypatch) -> None:
        from agent.config import CACHE_DIR_ENV, vectr_cache_root

        monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert vectr_cache_root() == tmp_path / ".cache" / "vectr"

    def test_reads_env_at_call_time_not_import_time(self, tmp_path, monkeypatch) -> None:
        """A fixture that sets the env var AFTER agent.config has already been
        imported (exactly what the session-scoped conftest fixture does) must
        still be honored -- vectr_cache_root must not cache a value at import."""
        from agent.config import CACHE_DIR_ENV, vectr_cache_root

        first = tmp_path / "first"
        second = tmp_path / "second"
        monkeypatch.setenv(CACHE_DIR_ENV, str(first))
        assert vectr_cache_root() == first
        monkeypatch.setenv(CACHE_DIR_ENV, str(second))
        assert vectr_cache_root() == second


class TestSessionWideIsolation:
    """The whole pytest session runs under the conftest.py autouse fixture --
    verify it is actually active and that real constructions land under it."""

    def test_session_cache_dir_env_var_is_set_and_not_the_real_cache(self) -> None:
        from agent.config import CACHE_DIR_ENV

        override = os.environ.get(CACHE_DIR_ENV)
        assert override, (
            f"{CACHE_DIR_ENV} must be set for the whole test session by "
            "conftest.py's _isolated_cache_root fixture"
        )
        assert Path(override) != _real_home_cache_root()

    def test_default_db_dir_resolves_under_isolated_root(self) -> None:
        from app.service import _default_db_dir
        from agent.config import vectr_cache_root

        db_dir = Path(_default_db_dir("/some/workspace/for/isolation/test"))
        assert db_dir.is_relative_to(vectr_cache_root())
        assert not db_dir.is_relative_to(_real_home_cache_root())

    def test_code_indexer_without_db_path_lands_under_isolated_root(
        self, tmp_path, monkeypatch
    ) -> None:
        """The exact construction pattern (CodeIndexer(workspace_root) with no
        explicit db_path) that minted the ~4,000 real-cache junk dirs a
        2026-07-20 cleanup swept -- must now resolve under the isolated root.

        The embed provider is stubbed because `CodeIndexer.__init__` builds one
        eagerly (agent/indexer/_core.py): with the session cache root isolated to
        a fresh tmp dir, a real provider finds nothing cached and downloads the
        model from Hugging Face on every run. That download is irrelevant to the
        db-path resolution under test and it made CI fail on an HF 429.
        """
        import agent.indexer as idx_module
        from agent.indexer import CodeIndexer
        from agent.config import vectr_cache_root
        from tests.conftest import _DummyEmbedProvider

        monkeypatch.setattr(
            idx_module, "get_embed_provider", lambda _model: _DummyEmbedProvider()
        )
        (tmp_path / "mod.py").write_text("def func_a(): pass")
        indexer = CodeIndexer(str(tmp_path))
        assert indexer._db_dir.is_relative_to(vectr_cache_root())
        assert not indexer._db_dir.is_relative_to(_real_home_cache_root())

    def test_model_cache_dirs_resolve_under_isolated_root(self) -> None:
        """The Hugging Face model cache dirs used by the embedder and
        reranker (agent/indexer/_types.py, agent/searcher.py) -- both must
        resolve under the isolated root too, not just the workspace DB dirs."""
        from agent.config import vectr_cache_root

        models_dir = vectr_cache_root() / "models"
        assert models_dir.is_relative_to(vectr_cache_root())
        assert not models_dir.is_relative_to(_real_home_cache_root())
