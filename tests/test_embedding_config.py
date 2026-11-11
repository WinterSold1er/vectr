"""Embed-model config layering on the 1.12.0 tree (UPG-EMBED-CONFIG-LAYERING).

The overlay lives in `agent/indexer/_types.py`: `_embed_config_for` merges
`<workspace>/.vectr/config.yaml` over `~/.vectr/config.yaml` (workspace wins
key by key), `resolve_embed_model` puts `VECTR_EMBED_MODEL` above both files
and above the bundled default, and `_google_output_dimensionality` reads
`VECTR_EMBED_OUTPUT_DIM` then `embedding.output_dimensionality`.

HOME and VECTR_WORKSPACE are pinned to tmp_path throughout, so the real
~/.vectr is never read or written.
"""
from __future__ import annotations

import pytest

import agent.indexer._types as types_mod
from agent.indexer._types import (
    _embed_config_for,
    _google_output_dimensionality,
    resolve_embed_model,
)

DEFAULT = "BUNDLED-DEFAULT"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """No real ~/.vectr, no real workspace, no ambient embedding env."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VECTR_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.delenv("VECTR_EMBED_MODEL", raising=False)
    monkeypatch.delenv("VECTR_EMBED_OUTPUT_DIM", raising=False)
    (tmp_path / "ws").mkdir()
    return tmp_path


def _write(path, model=None, dim=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "embedding:\n"
    if model is not None:
        body += f"  model: {model}\n"
    if dim is not None:
        body += f"  output_dimensionality: {dim}\n"
    path.write_text(body, encoding="utf-8")


def _global_file(tmp_path):
    return tmp_path / ".vectr" / "config.yaml"


def _workspace_file(tmp_path):
    return tmp_path / "ws" / ".vectr" / "config.yaml"


class TestOverlayLayering:
    def test_missing_overlays_yield_no_settings(self, isolated_env) -> None:
        # Nothing on disk: the caller keeps its own bundled default.
        assert _embed_config_for(None) == {}
        assert resolve_embed_model(None, None, DEFAULT) == DEFAULT

    def test_global_overlay_applies_when_workspace_has_none(self, isolated_env) -> None:
        _write(_global_file(isolated_env), model="global-model")
        assert resolve_embed_model(None, None, DEFAULT) == "global-model"

    def test_workspace_overlay_beats_global_overlay(self, isolated_env) -> None:
        _write(_global_file(isolated_env), model="global-model")
        _write(_workspace_file(isolated_env), model="workspace-model")
        assert resolve_embed_model(None, None, DEFAULT) == "workspace-model"

    def test_workspace_root_argument_overrides_the_env_pointer(self, isolated_env) -> None:
        _write(_global_file(isolated_env), model="global-model")
        _write(_workspace_file(isolated_env), model="workspace-model")
        other = isolated_env / "other"
        _write(other / ".vectr" / "config.yaml", model="explicit-root-model")
        assert resolve_embed_model(None, str(other), DEFAULT) == "explicit-root-model"

    def test_partial_overlay_keeps_earlier_values_it_does_not_set(self, isolated_env) -> None:
        _write(_global_file(isolated_env), model="global-model", dim=1536)
        _write(_workspace_file(isolated_env), model="workspace-model")
        cfg = _embed_config_for(None)
        assert cfg["model"] == "workspace-model"
        assert cfg["output_dimensionality"] == 1536

    def test_only_the_embedding_block_is_read(self, isolated_env) -> None:
        _global_file(isolated_env).parent.mkdir(parents=True, exist_ok=True)
        _global_file(isolated_env).write_text(
            "indexing:\n  max_file_size: 1\nembedding:\n  model: global-model\n",
            encoding="utf-8",
        )
        cfg = _embed_config_for(None)
        assert cfg == {"model": "global-model"}

    def test_overlay_without_an_embedding_key_is_a_no_op(self, isolated_env) -> None:
        _write(_global_file(isolated_env), model="global-model")
        ws = _workspace_file(isolated_env)
        ws.parent.mkdir(parents=True, exist_ok=True)
        ws.write_text("unrelated: 1\n", encoding="utf-8")
        # The workspace file says nothing about embedding, so the global one stands.
        assert resolve_embed_model(None, None, DEFAULT) == "global-model"

    def test_invalid_yaml_is_reported_and_names_the_file(self, isolated_env) -> None:
        bad = _workspace_file(isolated_env)
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("embedding: [unclosed\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="not valid YAML"):
            _embed_config_for(None)

    def test_unreadable_overlay_is_reported(self, isolated_env) -> None:
        target = _workspace_file(isolated_env)
        _write(target, model="workspace-model")
        target.chmod(0o000)  # owner-readable only is not enough: mode 0 is denied
        try:
            with pytest.raises(RuntimeError, match="could not be read"):
                _embed_config_for(None)
        finally:
            target.chmod(0o644)


class TestEnvironmentTier:
    def test_env_outranks_every_config_layer(self, isolated_env) -> None:
        _write(_global_file(isolated_env), model="global-model")
        _write(_workspace_file(isolated_env), model="workspace-model")
        assert resolve_embed_model("env-model", None, DEFAULT) == "env-model"

    def test_empty_env_falls_back_to_the_layered_config(self, isolated_env) -> None:
        _write(_workspace_file(isolated_env), model="workspace-model")
        assert resolve_embed_model("", None, DEFAULT) == "workspace-model"


class TestOutputDimensionality:
    def test_config_value_is_used_when_env_is_unset(self, isolated_env) -> None:
        _write(_workspace_file(isolated_env), dim=1536)
        assert _google_output_dimensionality(_embed_config_for(None)) == 1536

    def test_env_beats_the_config_value(self, isolated_env, monkeypatch) -> None:
        _write(_workspace_file(isolated_env), dim=1536)
        monkeypatch.setenv("VECTR_EMBED_OUTPUT_DIM", "3072")
        assert _google_output_dimensionality(_embed_config_for(None)) == 3072

    def test_unset_keeps_the_provider_default(self, isolated_env) -> None:
        assert _google_output_dimensionality({}) is None
        assert _google_output_dimensionality({"output_dimensionality": ""}) is None

    @pytest.mark.parametrize("raw", ["wide", 0, -8])
    def test_malformed_values_fail_loudly(self, isolated_env, monkeypatch, raw) -> None:
        monkeypatch.setenv("VECTR_EMBED_OUTPUT_DIM", str(raw))
        with pytest.raises(RuntimeError, match="output dimension"):
            _google_output_dimensionality({})
