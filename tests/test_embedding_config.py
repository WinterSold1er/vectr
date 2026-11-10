"""Tests for the layered embedding config (UPG-EMBED-CONFIG-LAYERING).

agent/config.py reads the bundled config.yaml once at import, so the layering
itself is exercised through `merge_embedding_overlays` / `_load_config` with
explicit tmp overlay files, and the environment tier through
`embed_model_spec()`. Every path here is a tmp_path: no real ~/.vectr and no
real workspace .vectr directory is read or written.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import agent.config as cfg

BUNDLED_MODEL = "ibm-granite/granite-embedding-english-r2"


def _bundled_cfg() -> dict:
    """The bundled `embedding:` block as _load_config() sees it before layering."""
    return {"embedding": {"default_model": BUNDLED_MODEL, "max_seq_length": 512}}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestBundledDefaults:
    def test_shipped_default_model_is_unchanged(self) -> None:
        assert cfg.EMBEDDING_DEFAULT_MODEL == BUNDLED_MODEL

    def test_bundled_yaml_ships_a_usable_embedding_block(self) -> None:
        assert cfg.EMBEDDING_OUTPUT_DIMENSIONALITY is None
        assert cfg.EMBEDDING_API_KEY_ENV == "GEMINI_API_KEY"
        assert cfg.EMBEDDING_MODEL_ENV == "VECTR_EMBED_MODEL"


class TestOverlayLayering:
    def test_missing_overlays_keep_bundled_values(self, tmp_path: Path) -> None:
        base = _bundled_cfg()
        cfg.merge_embedding_overlays(base, [tmp_path / "global.yaml", tmp_path / "ws.yaml"])
        assert base["embedding"]["default_model"] == BUNDLED_MODEL
        assert "output_dimensionality" not in base["embedding"]

    def test_global_overlay_applies_when_workspace_has_none(self, tmp_path: Path) -> None:
        global_file = _write(
            tmp_path / "global.yaml", "embedding:\n  model: google/gemini-embedding-2\n"
        )
        base = _bundled_cfg()
        cfg.merge_embedding_overlays(base, [global_file, tmp_path / "missing.yaml"])
        assert base["embedding"]["default_model"] == "google/gemini-embedding-2"

    def test_workspace_overlay_beats_global_overlay(self, tmp_path: Path) -> None:
        global_file = _write(tmp_path / "global.yaml", "embedding:\n  model: global/model\n")
        workspace_file = _write(tmp_path / "ws.yaml", "embedding:\n  model: workspace/model\n")
        base = _bundled_cfg()
        cfg.merge_embedding_overlays(base, [global_file, workspace_file])
        assert base["embedding"]["default_model"] == "workspace/model"

    def test_later_partial_overlay_keeps_earlier_values_it_does_not_set(
        self, tmp_path: Path
    ) -> None:
        global_file = _write(
            tmp_path
            / "global.yaml",
            "embedding:\n  model: global/model\n  output_dimensionality: 1536\n"
            "  api_key_env: GLOBAL_GEMINI_KEY\n",
        )
        workspace_file = _write(tmp_path / "ws.yaml", "embedding:\n  model: workspace/model\n")
        base = _bundled_cfg()
        cfg.merge_embedding_overlays(base, [global_file, workspace_file])
        assert base["embedding"]["default_model"] == "workspace/model"
        assert base["embedding"]["output_dimensionality"] == 1536
        assert base["embedding"]["api_key_env"] == "GLOBAL_GEMINI_KEY"

    def test_only_the_embedding_block_is_layered(self, tmp_path: Path) -> None:
        workspace_file = _write(
            tmp_path
            / "ws.yaml",
            "ranking:\n  quality_priors:\n    trivial: 0.99\nembedding:\n  model: m/x\n",
        )
        base = _bundled_cfg()
        cfg.merge_embedding_overlays(base, [workspace_file])
        assert base["embedding"]["default_model"] == "m/x"
        assert list(base) == ["embedding"], "non-embedding config must stay bundled-only"

    def test_overlay_without_an_embedding_key_is_a_no_op(self, tmp_path: Path) -> None:
        workspace_file = _write(tmp_path / "ws.yaml", "some_other_section:\n  a: 1\n")
        base = _bundled_cfg()
        cfg.merge_embedding_overlays(base, [workspace_file])
        assert base["embedding"]["default_model"] == BUNDLED_MODEL

    def test_invalid_yaml_is_reported_and_skipped(self, tmp_path: Path, caplog) -> None:
        bad_file = _write(tmp_path / "ws.yaml", "embedding:\n  model: [unclosed\n")
        base = _bundled_cfg()
        with caplog.at_level(logging.WARNING, logger="vectr.config"):
            cfg.merge_embedding_overlays(base, [bad_file])
        assert any("invalid YAML" in record.message for record in caplog.records)
        assert base["embedding"]["default_model"] == BUNDLED_MODEL

    def test_unreadable_overlay_is_reported_and_skipped(self, tmp_path: Path, caplog) -> None:
        not_a_file = _write(tmp_path / "ws.txt", "x")
        base = _bundled_cfg()
        with caplog.at_level(logging.WARNING, logger="vectr.config"):
            cfg.merge_embedding_overlays(base, [not_a_file / "config.yaml"])
        assert any("unreadable" in record.message for record in caplog.records)
        assert base["embedding"]["default_model"] == BUNDLED_MODEL


class TestOutputDimensionalityValidation:
    @pytest.mark.parametrize("raw", ["1536", 1536])
    def test_positive_values_are_accepted(self, raw) -> None:
        assert cfg._resolve_output_dimensionality(raw) == 1536

    def test_null_keeps_the_provider_default(self) -> None:
        assert cfg._resolve_output_dimensionality(None) is None

    @pytest.mark.parametrize("raw", ["wide", "0", 0, -8, True])
    def test_malformed_values_fail_loudly(self, raw) -> None:
        with pytest.raises(ValueError, match="output_dimensionality"):
            cfg._resolve_output_dimensionality(raw)


class TestFullLoadPrecedence:
    def test_load_config_layers_workspace_over_global(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("VECTR_WORKSPACE", str(tmp_path / "ws"))
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        _write(
            tmp_path / "home" / ".vectr" / "config.yaml",
            "embedding:\n  model: global/model\n  output_dimensionality: 768\n"
            "  api_key_env: GLOBAL_GEMINI_KEY\n",
        )
        _write(
            tmp_path / "ws" / ".vectr" / "config.yaml",
            "embedding:\n  model: google/gemini-embedding-2\n",
        )
        loaded = cfg._load_config()
        assert loaded["embedding"]["default_model"] == "google/gemini-embedding-2"
        assert loaded["embedding"]["output_dimensionality"] == 768
        assert loaded["embedding"]["api_key_env"] == "GLOBAL_GEMINI_KEY"
        assert loaded["embedding"]["max_seq_length"] == 512, "bundled keys survive"

    def test_overlay_paths_are_ordered_global_then_workspace(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("VECTR_WORKSPACE", str(tmp_path / "ws"))
        paths = cfg.embedding_overlay_paths()
        assert paths == [
            tmp_path / "home" / ".vectr" / "config.yaml",
            tmp_path / "ws" / ".vectr" / "config.yaml",
        ]

    def test_workspace_falls_back_to_cwd_when_env_absent(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("VECTR_WORKSPACE", raising=False)
        monkeypatch.chdir(tmp_path)
        assert cfg.embedding_overlay_paths()[-1] == tmp_path / ".vectr" / "config.yaml"


class TestEnvironmentTier:
    def test_env_outranks_every_config_layer(self, monkeypatch) -> None:
        monkeypatch.setenv("VECTR_EMBED_MODEL", "google/gemini-embedding-2")
        assert cfg.embed_model_spec() == "google/gemini-embedding-2"

    def test_empty_env_falls_back_to_the_layered_config(self, monkeypatch) -> None:
        monkeypatch.setenv("VECTR_EMBED_MODEL", "")
        assert cfg.embed_model_spec() == cfg.EMBEDDING_DEFAULT_MODEL

    def test_unset_env_uses_the_layered_config(self, monkeypatch) -> None:
        monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
        assert cfg.embed_model_spec() == cfg.EMBEDDING_DEFAULT_MODEL
