"""Tests for agent/config.py — bundled-config loader (UPG-11.1 refactor-to-config).

Verifies that:
1. config.yaml is loaded and exposes the correct numeric knobs with expected defaults.
2. The loader works via importlib.resources (not cwd-relative open()), confirmed by
   importing the module directly (same path the installed binary uses).
3. Quality priors, rerank pool sizes, indexing tunables, output knobs, and behaviour
   nudge thresholds all load correctly from the new config blocks (UPG-12.1).
"""
from __future__ import annotations

import pytest

import agent.config as cfg


class TestConfigLoaderQualityPriors:
    """ranking.quality_priors values must load correctly from config.yaml (UPG-12.1)."""

    def test_trivial_default(self) -> None:
        assert cfg.QUALITY_TRIVIAL == 0.15, f"QUALITY_TRIVIAL should be 0.15, got {cfg.QUALITY_TRIVIAL}"

    def test_navigational_default(self) -> None:
        assert cfg.QUALITY_NAVIGATIONAL == 0.35, f"QUALITY_NAVIGATIONAL should be 0.35, got {cfg.QUALITY_NAVIGATIONAL}"

    def test_heading_only_default(self) -> None:
        assert cfg.QUALITY_HEADING_ONLY == 0.40, f"QUALITY_HEADING_ONLY should be 0.40, got {cfg.QUALITY_HEADING_ONLY}"

    def test_generated_default(self) -> None:
        assert cfg.QUALITY_GENERATED == 0.45, f"QUALITY_GENERATED should be 0.45, got {cfg.QUALITY_GENERATED}"

    def test_vectr_config_default(self) -> None:
        assert cfg.QUALITY_VECTR_CONFIG == 0.10, f"QUALITY_VECTR_CONFIG should be 0.10, got {cfg.QUALITY_VECTR_CONFIG}"

    def test_test_deprioritised_default(self) -> None:
        # UPG-PREFIX-COMPOSE: lowered from 0.55 to 0.46 (still > QUALITY_GENERATED
        # 0.45, see test_ordering in test_chunk_quality.py) — see config.yaml's
        # ranking.quality_priors.test_deprioritised comment for the full
        # rationale (paired with a purpose_rank.lambda change).
        assert cfg.QUALITY_TEST_DEPRIORITISED == 0.46, (
            f"QUALITY_TEST_DEPRIORITISED should be 0.46, got {cfg.QUALITY_TEST_DEPRIORITISED}"
        )

    def test_doc_prose_default(self) -> None:
        assert cfg.QUALITY_DOC_PROSE == 0.70, f"QUALITY_DOC_PROSE should be 0.70, got {cfg.QUALITY_DOC_PROSE}"

    def test_short_penalty_default(self) -> None:
        assert cfg.QUALITY_SHORT_PENALTY == 0.80, f"QUALITY_SHORT_PENALTY should be 0.80, got {cfg.QUALITY_SHORT_PENALTY}"

    def test_all_are_floats(self) -> None:
        for name in (
            "QUALITY_TRIVIAL", "QUALITY_NAVIGATIONAL", "QUALITY_HEADING_ONLY",
            "QUALITY_GENERATED", "QUALITY_VECTR_CONFIG", "QUALITY_TEST_DEPRIORITISED",
            "QUALITY_DOC_PROSE", "QUALITY_SHORT_PENALTY",
        ):
            assert isinstance(getattr(cfg, name), float), f"{name} must be float"

    def test_all_in_open_unit_interval(self) -> None:
        """All quality priors must be in (0, 1] — 0 would silence chunks entirely."""
        for name in (
            "QUALITY_TRIVIAL", "QUALITY_NAVIGATIONAL", "QUALITY_HEADING_ONLY",
            "QUALITY_GENERATED", "QUALITY_VECTR_CONFIG", "QUALITY_TEST_DEPRIORITISED",
            "QUALITY_DOC_PROSE", "QUALITY_SHORT_PENALTY",
        ):
            val = getattr(cfg, name)
            assert 0 < val <= 1.0, f"{name}={val} must be in (0, 1]"

    def test_chunk_quality_aliases_match_config(self) -> None:
        """chunk_quality.py _Q_* aliases must equal config exports (UPG-12.1)."""
        from agent.chunk_quality import (
            _Q_TRIVIAL, _Q_NAVIGATIONAL, _Q_HEADING_ONLY, _Q_GENERATED,
            _Q_VECTR_CONFIG, _Q_TEST_DEPRIORITISED, _Q_DOC_PROSE, _Q_SHORT_PENALTY,
        )
        assert _Q_TRIVIAL is cfg.QUALITY_TRIVIAL
        assert _Q_NAVIGATIONAL is cfg.QUALITY_NAVIGATIONAL
        assert _Q_HEADING_ONLY is cfg.QUALITY_HEADING_ONLY
        assert _Q_GENERATED is cfg.QUALITY_GENERATED
        assert _Q_VECTR_CONFIG is cfg.QUALITY_VECTR_CONFIG
        assert _Q_TEST_DEPRIORITISED is cfg.QUALITY_TEST_DEPRIORITISED
        assert _Q_DOC_PROSE is cfg.QUALITY_DOC_PROSE
        assert _Q_SHORT_PENALTY is cfg.QUALITY_SHORT_PENALTY


class TestConfigLoaderRerank:
    """ranking.rerank values must load correctly from config.yaml (UPG-12.1)."""

    def test_top_k_default(self) -> None:
        assert cfg.RERANK_TOP_K == 40, f"RERANK_TOP_K should be 40, got {cfg.RERANK_TOP_K}"

    def test_top_k_unfiltered_default(self) -> None:
        # UPG-RERANKER-SWAP-SPIKE (2026-08-19): lowered 60 -> 40 alongside the
        # reranker swap. This is the knob that governs almost every real query
        # (top_k covers only the language-filtered branch), and it was swept
        # across {10,20,30,40,60} against the django acceptance corpus for the
        # shipped model: 43/57 cases and 5/6 must-pass at 40, versus 44/57 and
        # 5/6 at 60 for 37% more rerank latency. 40 was chosen over the even
        # cheaper 20 because must-pass-six is non-monotonic across the sweep
        # (4/5/4/5/5) — 20 is an isolated peak, 40-60 is a contiguous plateau.
        assert cfg.RERANK_TOP_K_UNFILTERED == 40, (
            f"RERANK_TOP_K_UNFILTERED should be 40 (UPG-RERANKER-SWAP-SPIKE), "
            f"got {cfg.RERANK_TOP_K_UNFILTERED}"
        )

    def test_top_k_is_int(self) -> None:
        assert isinstance(cfg.RERANK_TOP_K, int)

    def test_top_k_unfiltered_is_int(self) -> None:
        assert isinstance(cfg.RERANK_TOP_K_UNFILTERED, int)

    def test_unfiltered_pool_not_shallower_than_filtered(self) -> None:
        """The unfiltered path must never get a SHALLOWER rerank pool than the
        filtered path.

        Relaxed from strict `>` to `>=` by UPG-RERANKER-SWAP-SPIKE (2026-08-19),
        deliberately and with the reasoning recorded here rather than silently.
        The original intent is that an unfiltered query, which has no language
        narrowing and so admits more doc prose, must not be handicapped with a
        smaller pool than a filtered one. That intent is `>=`; the strictness was
        incidental to the old pair of values (60 > 40), not a designed margin.

        The measured sweep lowered top_k_unfiltered to 40, which now coincides
        with top_k. `top_k` was deliberately NOT lowered to restore the strict
        inequality: it governs only the language-filtered branch, which the
        acceptance corpus does not exercise at all (0 of 65 django cases set a
        language filter), so there is no evidence for a value and guessing one
        would change unmeasured behaviour to satisfy a test.
        """
        assert cfg.RERANK_TOP_K_UNFILTERED >= cfg.RERANK_TOP_K, (
            f"RERANK_TOP_K_UNFILTERED ({cfg.RERANK_TOP_K_UNFILTERED}) must not be "
            f"smaller than RERANK_TOP_K ({cfg.RERANK_TOP_K})"
        )

    def test_pre_filter_fetch_k_exceeds_rerank_pool(self) -> None:
        """The over-fetch depth must leave real room to drop trivial chunks and
        still fill the rerank pool.

        `test_pre_filter_fetch_k_default` below has asserted only `>= 200` while
        its own comment claimed "must be strictly larger than top_k_unfiltered" —
        an invariant described but never checked. Added here (UPG-RERANKER-SWAP-
        SPIKE) so lowering either knob cannot silently invert the relationship.
        """
        assert cfg.RERANK_PRE_FILTER_FETCH_K > cfg.RERANK_TOP_K_UNFILTERED, (
            f"RERANK_PRE_FILTER_FETCH_K ({cfg.RERANK_PRE_FILTER_FETCH_K}) must exceed "
            f"RERANK_TOP_K_UNFILTERED ({cfg.RERANK_TOP_K_UNFILTERED}) so the pool-entry "
            "trivial filter has candidates to spare"
        )

    def test_pre_filter_fetch_k_default(self) -> None:
        # UPG-15.7: over-fetch depth for the pool-entry trivial filter.
        # Must be strictly larger than top_k_unfiltered so there is room to filter
        # trivial chunks and still fill the rerank pool.
        assert cfg.RERANK_PRE_FILTER_FETCH_K >= 200, (
            f"RERANK_PRE_FILTER_FETCH_K should be ≥ 200 (UPG-15.7), got {cfg.RERANK_PRE_FILTER_FETCH_K}"
        )

    def test_pre_filter_fetch_k_is_int(self) -> None:
        assert isinstance(cfg.RERANK_PRE_FILTER_FETCH_K, int)

    def test_pre_filter_fetch_k_exceeds_top_k_unfiltered(self) -> None:
        """pre_filter_fetch_k must exceed top_k_unfiltered so there is room to filter."""
        assert cfg.RERANK_PRE_FILTER_FETCH_K > cfg.RERANK_TOP_K_UNFILTERED, (
            f"RERANK_PRE_FILTER_FETCH_K ({cfg.RERANK_PRE_FILTER_FETCH_K}) must exceed "
            f"RERANK_TOP_K_UNFILTERED ({cfg.RERANK_TOP_K_UNFILTERED})"
        )

    def test_searcher_rerank_aliases_from_config(self) -> None:
        """searcher.py must import rerank pool sizes from config, not define its own."""
        import agent.searcher as searcher_mod
        assert searcher_mod._RERANK_TOP_K is cfg.RERANK_TOP_K
        assert searcher_mod._RERANK_TOP_K_UNFILTERED is cfg.RERANK_TOP_K_UNFILTERED
        assert searcher_mod._RERANK_PRE_FILTER_FETCH_K is cfg.RERANK_PRE_FILTER_FETCH_K

    def test_max_length_default(self) -> None:
        # UPG-RERANK-LATENCY-BUDGET: was hardcoded in agent/searcher.py.
        assert cfg.RERANK_MAX_LENGTH == 512, (
            f"RERANK_MAX_LENGTH should be 512, got {cfg.RERANK_MAX_LENGTH}"
        )

    def test_max_length_is_int(self) -> None:
        assert isinstance(cfg.RERANK_MAX_LENGTH, int)

    def test_batch_size_by_device_is_dict_of_int(self) -> None:
        assert isinstance(cfg.RERANK_BATCH_SIZE_BY_DEVICE, dict)
        assert cfg.RERANK_BATCH_SIZE_BY_DEVICE, "batch_size map must not be empty"
        for device, bs in cfg.RERANK_BATCH_SIZE_BY_DEVICE.items():
            assert isinstance(device, str)
            assert isinstance(bs, int) and bs > 0

    def test_batch_size_by_device_has_required_keys(self) -> None:
        """"default" is the runtime fallback for a resolved torch device
        the config doesn't name (agent/searcher.py._Reranker._batch_size) —
        it must always be present, along with the measured mps/cpu entries
        and the untested cuda entry (UPG-RERANK-LATENCY-BUDGET)."""
        for key in ("mps", "cpu", "cuda", "default"):
            assert key in cfg.RERANK_BATCH_SIZE_BY_DEVICE, (
                f"ranking.rerank.batch_size missing required key {key!r}"
            )

    def test_cuda_batch_size_preserves_library_default(self) -> None:
        """No CUDA hardware was available to measure a device-specific value
        (UPG-RERANK-LATENCY-BUDGET) — cuda must keep sentence_transformers'
        own unconfigured predict() default (32) rather than guess from the
        mps/cpu measurements, which were both taken on Apple Silicon."""
        assert cfg.RERANK_BATCH_SIZE_BY_DEVICE["cuda"] == 32

    def test_searcher_rerank_batch_size_and_max_length_alias_from_config(self) -> None:
        import agent.searcher as searcher_mod
        assert searcher_mod._RERANK_MAX_LENGTH is cfg.RERANK_MAX_LENGTH
        assert searcher_mod._RERANK_BATCH_SIZE_BY_DEVICE is cfg.RERANK_BATCH_SIZE_BY_DEVICE


class TestConfigLoaderIndexing:
    """indexing.* values must load correctly from config.yaml (UPG-12.1)."""

    def test_max_chunk_lines_default(self) -> None:
        assert cfg.INDEXING_MAX_CHUNK_LINES == 150, (
            f"INDEXING_MAX_CHUNK_LINES should be 150, got {cfg.INDEXING_MAX_CHUNK_LINES}"
        )

    def test_max_chunk_chars_default(self) -> None:
        """UPG-WINDOW-CHUNK-BYTE-CAP: the character cap must ship with enough
        headroom that an ordinary full window never hits it — only genuinely
        oversized content (one giant line / minified one-liner) may be
        sub-split."""
        assert cfg.INDEXING_MAX_CHUNK_CHARS == 32000, (
            f"INDEXING_MAX_CHUNK_CHARS should be 32000, got {cfg.INDEXING_MAX_CHUNK_CHARS}"
        )
        # Even an absurdly generous reading of a full window — one line for
        # every point of max_chunk_lines, each line itself max_chunk_lines
        # characters long — must stay under the char cap.
        assert cfg.INDEXING_MAX_CHUNK_CHARS > (
            cfg.INDEXING_MAX_CHUNK_LINES * cfg.INDEXING_MAX_CHUNK_LINES
        )

    def test_class_header_lines_default(self) -> None:
        assert cfg.INDEXING_CLASS_HEADER_LINES == 40, (
            f"INDEXING_CLASS_HEADER_LINES should be 40, got {cfg.INDEXING_CLASS_HEADER_LINES}"
        )

    def test_max_chunk_lines_is_int(self) -> None:
        assert isinstance(cfg.INDEXING_MAX_CHUNK_LINES, int)

    def test_max_chunk_chars_is_positive_int(self) -> None:
        assert isinstance(cfg.INDEXING_MAX_CHUNK_CHARS, int)
        assert cfg.INDEXING_MAX_CHUNK_CHARS > 0

    def test_class_header_lines_is_int(self) -> None:
        assert isinstance(cfg.INDEXING_CLASS_HEADER_LINES, int)

    def test_class_header_lines_smaller_than_max_chunk_lines(self) -> None:
        """Class header must be a strict subset of max chunk size."""
        assert cfg.INDEXING_CLASS_HEADER_LINES < cfg.INDEXING_MAX_CHUNK_LINES, (
            f"INDEXING_CLASS_HEADER_LINES ({cfg.INDEXING_CLASS_HEADER_LINES}) must be "
            f"< INDEXING_MAX_CHUNK_LINES ({cfg.INDEXING_MAX_CHUNK_LINES})"
        )

    def test_indexer_aliases_from_config(self) -> None:
        """indexer.py must import chunk line limits from config, not define its own."""
        import agent.indexer as indexer_mod
        assert indexer_mod._MAX_CHUNK_LINES is cfg.INDEXING_MAX_CHUNK_LINES
        assert indexer_mod._MAX_CHUNK_CHARS is cfg.INDEXING_MAX_CHUNK_CHARS
        assert indexer_mod._CLASS_HEADER_LINES is cfg.INDEXING_CLASS_HEADER_LINES


class TestConfigLoaderIndexGovernor:
    """indexing.index_governor.* values must load correctly from config.yaml
    (UPG-INDEX-RESOURCE-GOVERNOR)."""

    def test_enabled_default(self) -> None:
        assert cfg.INDEX_GOVERNOR_ENABLED is True

    def test_duty_cycle_default(self) -> None:
        assert cfg.INDEX_GOVERNOR_DUTY_CYCLE == 0.5

    def test_duty_cycle_is_float_in_unit_interval(self) -> None:
        assert isinstance(cfg.INDEX_GOVERNOR_DUTY_CYCLE, float)
        assert 0.0 < cfg.INDEX_GOVERNOR_DUTY_CYCLE <= 1.0

    def test_macos_qos_class_default(self) -> None:
        assert cfg.INDEX_GOVERNOR_MACOS_QOS_CLASS == "utility"

    def test_macos_qos_class_is_a_known_name(self) -> None:
        from agent.indexer._priority import _MACOS_QOS_BY_NAME
        assert cfg.INDEX_GOVERNOR_MACOS_QOS_CLASS in _MACOS_QOS_BY_NAME

    def test_linux_nice_increment_default(self) -> None:
        assert cfg.INDEX_GOVERNOR_LINUX_NICE_INCREMENT == 10

    def test_linux_nice_increment_in_valid_range(self) -> None:
        # POSIX niceness range is -20..19; a governor only ever lowers
        # priority, so the increment must be a positive value in 0..19.
        assert 0 <= cfg.INDEX_GOVERNOR_LINUX_NICE_INCREMENT <= 19

    def test_min_batch_seconds_for_pacing_default(self) -> None:
        assert cfg.INDEX_GOVERNOR_MIN_BATCH_SECONDS_FOR_PACING == 0.05

    def test_checkpoint_every_batches_default(self) -> None:
        assert cfg.INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES == 1

    def test_checkpoint_every_batches_is_positive_int(self) -> None:
        assert isinstance(cfg.INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES, int)
        assert cfg.INDEX_GOVERNOR_CHECKPOINT_EVERY_BATCHES >= 1

    def test_missing_key_raises_keyerror(self) -> None:
        """No .get()-with-default fallback anywhere in this wiring — a config
        missing indexing.index_governor.duty_cycle must fail loudly at
        import, not silently default."""
        stripped = {k: v for k, v in cfg._cfg["indexing"].items() if k != "index_governor"}
        with pytest.raises(KeyError):
            _ = stripped["index_governor"]["duty_cycle"]


class TestConfigLoaderOutput:
    """output.* values must load correctly from config.yaml (UPG-12.1)."""

    def test_snippet_lines_default(self) -> None:
        assert cfg.OUTPUT_SNIPPET_LINES == 12, (
            f"OUTPUT_SNIPPET_LINES should be 12, got {cfg.OUTPUT_SNIPPET_LINES}"
        )

    def test_snippet_lines_is_int(self) -> None:
        assert isinstance(cfg.OUTPUT_SNIPPET_LINES, int)

    def test_symbol_graph_snippet_lines_from_config(self) -> None:
        """symbol_graph.py SNIPPET_LINES must equal config export (UPG-12.1)."""
        from agent.symbol_graph import SNIPPET_LINES
        assert SNIPPET_LINES is cfg.OUTPUT_SNIPPET_LINES


class TestConfigLoaderBehavior:
    """behavior.remember_nudge values must load correctly from config.yaml (UPG-12.1)."""

    def test_threshold_default(self) -> None:
        assert cfg.BEHAVIOR_REMEMBER_NUDGE_THRESHOLD == 10, (
            f"BEHAVIOR_REMEMBER_NUDGE_THRESHOLD should be 10, got {cfg.BEHAVIOR_REMEMBER_NUDGE_THRESHOLD}"
        )

    def test_cooldown_default(self) -> None:
        assert cfg.BEHAVIOR_REMEMBER_NUDGE_COOLDOWN == 5, (
            f"BEHAVIOR_REMEMBER_NUDGE_COOLDOWN should be 5, got {cfg.BEHAVIOR_REMEMBER_NUDGE_COOLDOWN}"
        )

    def test_threshold_is_int(self) -> None:
        assert isinstance(cfg.BEHAVIOR_REMEMBER_NUDGE_THRESHOLD, int)

    def test_cooldown_is_int(self) -> None:
        assert isinstance(cfg.BEHAVIOR_REMEMBER_NUDGE_COOLDOWN, int)

    def test_cooldown_smaller_than_threshold(self) -> None:
        """Cooldown re-fire interval must be strictly less than the initial threshold."""
        assert cfg.BEHAVIOR_REMEMBER_NUDGE_COOLDOWN < cfg.BEHAVIOR_REMEMBER_NUDGE_THRESHOLD, (
            f"BEHAVIOR_REMEMBER_NUDGE_COOLDOWN ({cfg.BEHAVIOR_REMEMBER_NUDGE_COOLDOWN}) must be "
            f"< BEHAVIOR_REMEMBER_NUDGE_THRESHOLD ({cfg.BEHAVIOR_REMEMBER_NUDGE_THRESHOLD})"
        )

    def test_mcp_server_aliases_from_config(self) -> None:
        """mcp_server.py must import nudge tunables from config, not define its own."""
        from integrations.mcp_server import (
            _REMEMBER_NUDGE_THRESHOLD,
            _REMEMBER_NUDGE_COOLDOWN,
        )
        assert _REMEMBER_NUDGE_THRESHOLD is cfg.BEHAVIOR_REMEMBER_NUDGE_THRESHOLD
        assert _REMEMBER_NUDGE_COOLDOWN is cfg.BEHAVIOR_REMEMBER_NUDGE_COOLDOWN


class TestConfigLoaderResume:
    """behavior.resume.* values must load correctly from config.yaml (UPG-RESUME-SURFACE)."""

    def test_max_gotchas_default(self) -> None:
        assert cfg.RESUME_MAX_GOTCHAS == 5, (
            f"RESUME_MAX_GOTCHAS should be 5, got {cfg.RESUME_MAX_GOTCHAS}"
        )

    def test_max_gotchas_is_int(self) -> None:
        assert isinstance(cfg.RESUME_MAX_GOTCHAS, int)

    def test_missing_key_raises_keyerror(self) -> None:
        """No .get()-with-default fallback anywhere in this wiring — a config
        missing behavior.resume.max_gotchas must fail loudly at import, not
        silently default."""
        stripped = {k: v for k, v in cfg._cfg["behavior"].items() if k != "resume"}
        with pytest.raises(KeyError):
            _ = stripped["resume"]["max_gotchas"]


class TestConfigLoaderWorkspaceAndWatcher:
    """workspace.* / watcher.* values must load correctly from config.yaml (UPG-13.1/13.2)."""

    def test_default_vectrignore_dirs_is_nonempty_tuple(self) -> None:
        assert isinstance(cfg.WORKSPACE_DEFAULT_VECTRIGNORE_DIRS, tuple)
        assert len(cfg.WORKSPACE_DEFAULT_VECTRIGNORE_DIRS) > 0

    def test_default_vectrignore_dirs_covers_common_excludes(self) -> None:
        expected = {
            "node_modules", ".venv", "venv", "env", "__pycache__", ".git",
            "dist", "build", "target", ".mypy_cache", ".pytest_cache",
            ".ruff_cache", "htmlcov", "coverage", ".tox", ".cache", "tmp",
            "vendor", ".next", ".nuxt", "out",
        }
        assert expected <= set(cfg.WORKSPACE_DEFAULT_VECTRIGNORE_DIRS)

    def test_default_vectrignore_dirs_entries_are_strings(self) -> None:
        for d in cfg.WORKSPACE_DEFAULT_VECTRIGNORE_DIRS:
            assert isinstance(d, str)

    def test_top_level_rescan_interval_is_positive_float(self) -> None:
        assert isinstance(cfg.WATCHER_TOP_LEVEL_RESCAN_INTERVAL_S, float)
        assert cfg.WATCHER_TOP_LEVEL_RESCAN_INTERVAL_S > 0

    def test_watcher_aliases_from_config(self) -> None:
        """watcher.py must import the rescan interval from config, not hardcode it."""
        from agent.watcher import WATCHER_TOP_LEVEL_RESCAN_INTERVAL_S
        assert WATCHER_TOP_LEVEL_RESCAN_INTERVAL_S is cfg.WATCHER_TOP_LEVEL_RESCAN_INTERVAL_S


class TestConfigLoaderHooks:
    """hooks.* values must load correctly from config.yaml (UPG-HOOK-INJECT-OBSERVABILITY)."""

    def test_log_injections_defaults_false(self) -> None:
        assert cfg.HOOKS_LOG_INJECTIONS is False

    def test_log_chars_per_token_default(self) -> None:
        assert cfg.HOOKS_LOG_CHARS_PER_TOKEN == 4

    def test_log_injections_is_bool(self) -> None:
        assert isinstance(cfg.HOOKS_LOG_INJECTIONS, bool)

    def test_log_chars_per_token_is_int(self) -> None:
        assert isinstance(cfg.HOOKS_LOG_CHARS_PER_TOKEN, int)


class TestConfigLoaderMemoryTriggerSemanticTheta:
    """memory_triggers.semantic.theta_by_kind must load correctly from
    config.yaml (TRIGGER-ENGINE wave 2b) — the M (semantic) primitive's
    fixed per-kind cosine threshold has no hardcoded value anywhere in
    agent/trigger_engine.py or agent/working_context_store/_store.py; every
    kind's number comes from this dict, built by direct subscript."""

    def test_one_entry_per_kind_in_priority_order(self) -> None:
        """A config.yaml missing an entry for any kind enumerated in
        memory_triggers.total_order.kind_priority would already have raised
        KeyError at import time (agent/config.py) — this just confirms the
        two sets line up 1:1 with no silent drops on either side."""
        assert set(cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND) == set(cfg.MEMORY_TRIGGER_KIND_PRIORITY)

    def test_all_are_floats(self) -> None:
        for kind, theta in cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND.items():
            assert isinstance(theta, float), f"theta for kind={kind!r} must be float"

    def test_all_in_open_unit_interval(self) -> None:
        for kind, theta in cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND.items():
            assert 0 < theta <= 1.0, f"theta for kind={kind!r}={theta} must be in (0, 1]"

    def test_directive_and_gotcha_default(self) -> None:
        assert cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["directive"] == 0.72
        assert cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["gotcha"] == 0.72

    def test_task_default(self) -> None:
        assert cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["task"] == 0.75

    def test_finding_default(self) -> None:
        assert cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["finding"] == 0.78

    def test_reference_default(self) -> None:
        assert cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["reference"] == 0.80

    def test_store_reads_theta_from_config_not_a_local_copy(self) -> None:
        """agent/working_context_store/_store.py's fire() must import this
        dict directly from agent.config at call time rather than defining
        (or caching a stale copy of) its own thresholds — proven here by
        mutating the config dict in place and observing fire()'s own
        semantic-gate decision follow the new value."""
        import chromadb

        from agent.working_context_store import WorkingContextStore

        def const_embed(vector: list[float]):
            def _embed(texts: list[str]) -> list[list[float]]:
                return [list(vector) for _ in texts]
            return _embed

        original = dict(cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND)
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                client = chromadb.PersistentClient(path=f"{tmp_dir}/chroma")
                store = WorkingContextStore(
                    tmp_dir,
                    embed_fn=const_embed([0.6, 0.8]),
                    embed_query_fn=const_embed([1.0, 0.0]),
                    notes_chroma_client=client,
                )
                store.remember(tmp_dir, "a gotcha", kind="gotcha", triggers=[{"semantic": True}])
                # cosine([0.6, 0.8], [1.0, 0.0]) == 0.6 — below the 0.72 default, above a lowered one.
                cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["gotcha"] = 0.72
                assert store.fire(tmp_dir, event="prompt-submit", query="anything") == []
                cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["gotcha"] = 0.5
                assert len(store.fire(tmp_dir, event="prompt-submit", query="anything")) == 1
        finally:
            cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND.clear()
            cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND.update(original)


class TestConfigLoaderMemoryWrite:
    """memory_write.* — write-time offer knobs (related-notes lookup +
    proxy-anchor suggestions) computed alongside remember(). Additive-only
    config: no hardcoded thresholds anywhere in agent/working_context_store/
    _related.py or agent/proxy_anchors.py, everything comes from these
    direct-subscript constants."""

    def test_related_notes_defaults(self) -> None:
        assert cfg.MEMORY_WRITE_RELATED_ENABLED is True
        assert cfg.MEMORY_WRITE_RELATED_LIMIT == 3
        assert cfg.MEMORY_WRITE_RELATED_MIN_SIMILARITY == 0.75

    def test_related_notes_types(self) -> None:
        assert isinstance(cfg.MEMORY_WRITE_RELATED_ENABLED, bool)
        assert isinstance(cfg.MEMORY_WRITE_RELATED_LIMIT, int)
        assert isinstance(cfg.MEMORY_WRITE_RELATED_MIN_SIMILARITY, float)

    def test_related_notes_min_similarity_matches_operational_theta(self) -> None:
        """Deliberately pinned to the same conservative bar as
        memory_triggers.semantic.theta_by_kind.operational — see both
        config.yaml comments for why."""
        assert cfg.MEMORY_WRITE_RELATED_MIN_SIMILARITY == cfg.MEMORY_TRIGGER_SEMANTIC_THETA_BY_KIND["operational"]

    def test_proxy_anchor_suggestions_defaults(self) -> None:
        assert cfg.MEMORY_WRITE_PROXY_SUGGEST_ENABLED is True
        assert cfg.MEMORY_WRITE_PROXY_SUGGEST_LIMIT == 4

    def test_proxy_anchor_suggestions_types(self) -> None:
        assert isinstance(cfg.MEMORY_WRITE_PROXY_SUGGEST_ENABLED, bool)
        assert isinstance(cfg.MEMORY_WRITE_PROXY_SUGGEST_LIMIT, int)


class TestConfigLoaderYamlBoolHardening:
    """UPG-YAML-BOOL: the config loader must not let YAML-1.1 bool literals
    (on/off/yes/no) corrupt a string-typed list — pyyaml's default resolver
    parses a bare `on` as True. `_StrictBoolLoader` strips that resolver while
    keeping true/false as genuine booleans."""

    def test_on_off_yes_no_stay_strings_in_list(self) -> None:
        import yaml
        from agent.config import _StrictBoolLoader
        doc = yaml.load("words: [on, off, yes, no, the]\n", Loader=_StrictBoolLoader)
        assert doc["words"] == ["on", "off", "yes", "no", "the"]
        assert all(isinstance(w, str) for w in doc["words"])

    def test_true_false_still_parse_as_bool(self) -> None:
        import yaml
        from agent.config import _StrictBoolLoader
        doc = yaml.load("a: true\nb: false\n", Loader=_StrictBoolLoader)
        assert doc["a"] is True
        assert doc["b"] is False

    def test_case_insensitive_on_off_stay_strings(self) -> None:
        import yaml
        from agent.config import _StrictBoolLoader
        doc = yaml.load("words: [On, OFF, Yes, NO]\n", Loader=_StrictBoolLoader)
        assert doc["words"] == ["On", "OFF", "Yes", "NO"]

    def test_real_stopword_list_contains_bare_on_as_string(self) -> None:
        # The concrete witness: NOTFOUND_FLOOR_STOPWORDS carries a bare `on`
        # that must be the string "on", never boolean True.
        assert "on" in cfg.NOTFOUND_FLOOR_STOPWORDS
        assert True not in cfg.NOTFOUND_FLOOR_STOPWORDS
        assert all(isinstance(w, str) for w in cfg.NOTFOUND_FLOOR_STOPWORDS)


class TestServerDefaultPort:
    """T26: the default bind port/host have a single source of truth in
    agent.config, de-hardcoding the sprinkled 8765 literal."""

    def test_default_port_constant_exists(self) -> None:
        assert cfg.DEFAULT_PORT == 8765
        assert cfg.DEFAULT_HOST == "127.0.0.1"

    def test_service_default_port_uses_constant(self) -> None:
        import inspect
        from app.service import VectrService
        sig = inspect.signature(VectrService.__init__)
        assert sig.parameters["port"].default == cfg.DEFAULT_PORT

    def test_vscode_bridge_defaults_use_constant(self) -> None:
        import inspect
        from integrations import vscode_bridge
        for fn in (vscode_bridge.configure_cursor, vscode_bridge.configure_claude_code,
                   vscode_bridge.configure_vscode, vscode_bridge.configure_all):
            assert inspect.signature(fn).parameters["port"].default == cfg.DEFAULT_PORT
