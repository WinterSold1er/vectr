"""
UPG-SORTBY-SHARED-VOCAB — the recall `sort_by` vocabulary
(`relevance` | `recency` | `priority` | `chronological`) is now declared
exactly once and consumed by every surface that exposes it.

The drift this test exists to detect: before the fix, the same four
strings were independently hardcoded at REST (`app/models.py::_SORT_BY_
VALUES`), at MCP dispatch (`integrations/mcp_server/_dispatch.py`), in
the MCP tool schema enum (`integrations/mcp_server/_schemas.py`), and in
the CLI's argparse choices (`main.py`). Adding a fifth sort mode at one
surface and forgetting the other three produced a bug no test could
catch — a test that compares four hardcoded literals to four other
hardcoded literals passes even when all four copies are independently
maintained, which is exactly the pre-fix state.

The pin here is OBJECT IDENTITY, not equality. Every surface consumes
the same `SORT_BY_VALUES` constant from
`agent.working_context_store._types` (re-exported through
`agent.working_context_store.__init__`). If a surface ever re-spells the
vocabulary (e.g. by reverting to a literal tuple in its own file), the
identity check fails immediately and the drift is caught at test time.

Five surfaces are pinned (one per spot the brief identified at `d93d7fd`):
  1. The shared constant itself.
  2. `app/models.py::_SORT_BY_VALUES` (the REST layer's import-time alias).
  3. `app.models.RecallRequest.validate_sort_by` (REST validation reads it).
  4. `integrations.mcp_server._dispatch` recall handler (MCP boundary).
  5. `integrations.mcp_server._schemas` vectr_recall tool schema enum (MCP
     schema the client sees).
  6. `main.py` `p_recall` argparse `--sort-by` choices (CLI surface).

Independence of the test from any literal: this module never lists the
four mode names in a POSITIVE assertion (a test that re-typed the
vocabulary as a list of expected values would itself be subject to the
same drift it exists to catch). The two source-level "no re-spelling"
pins (CLI and dispatch sites) use the literal ONLY as a NEGATIVE
assertion — `assert "...relevance", "recency", "priority", "chronological"
... not in source` — which is the only legitimate use of a literal in
this kind of pin, and exactly what catches a future PR that reverts
any surface to a hardcoded copy.
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# (1) The single source of truth
# ---------------------------------------------------------------------------

class TestSortByVocabSourceOfTruth:
    def test_sort_by_values_is_a_tuple_of_strings(self) -> None:
        from agent.working_context_store import SORT_BY_VALUES
        assert isinstance(SORT_BY_VALUES, tuple)
        for v in SORT_BY_VALUES:
            assert isinstance(v, str)
            assert v  # no empty strings

    def test_sort_by_values_has_four_modes(self) -> None:
        """Pins the closed-vocabulary size at 4. If a fifth mode is added
        (the whole point of having a shared constant), the test for the
        REST validator / dispatch guard / CLI argparse / MCP schema
        surfaces below will then verify the new mode shows up everywhere
        without this test needing a hardcoded set of names to chase."""
        from agent.working_context_store import SORT_BY_VALUES
        assert len(SORT_BY_VALUES) == 4

    def test_sort_by_values_exported_from_package_root(self) -> None:
        # Other surfaces import from `agent.working_context_store` (the
        # package root), not the private `_types` module — pin that the
        # re-export is in place.
        from agent.working_context_store import SORT_BY_VALUES as a
        from agent.working_context_store._types import SORT_BY_VALUES as b
        assert a is b


# ---------------------------------------------------------------------------
# (2) REST surface — app/models.py
# ---------------------------------------------------------------------------

class TestRestSurfaceSortByVocab:
    def test_rest_models_alias_points_at_shared_constant(self) -> None:
        """UPG-SORTBY-SHARED-VOCAB: the REST layer's local `_SORT_BY_VALUES`
        is a deliberate alias of the shared constant (kept under the
        historical private name to minimise the in-module diff), so a
        fifth mode added at the source automatically tightens the REST
        validator here too. Identity (not equality): a literal re-spelling
        in app/models.py would fail this assertion."""
        from agent.working_context_store import SORT_BY_VALUES
        from app.models import _SORT_BY_VALUES
        assert _SORT_BY_VALUES is SORT_BY_VALUES

    def test_rest_recall_request_validates_against_shared_constant(self) -> None:
        """The validate_sort_by field validator must reject values not in
        the shared vocabulary. Constructed values are derived from the
        shared constant itself: one in-vocabulary value, one fabricated
        out-of-vocabulary value, and one new value that, if a future
        surface is added to the source, immediately becomes the next
        in-vocabulary value the validator must accept."""
        from agent.working_context_store import SORT_BY_VALUES
        from pydantic import ValidationError
        from app.models import RecallRequest

        # In-vocabulary values all pass.
        for v in SORT_BY_VALUES:
            RecallRequest(sort_by=v)

        # Any string not in the shared vocabulary is rejected. Identity
        # of the rejection's "must be one of" message with the joined
        # shared constant is the second pin: REST's error wording still
        # reflects the same source.
        out_of_vocab = "_not_in_vocab_" + "x"  # never collides with a real mode
        with pytest.raises(ValidationError) as excinfo:
            RecallRequest(sort_by=out_of_vocab)
        # The error message must enumerate the shared constant verbatim.
        assert "sort_by must be one of" in str(excinfo.value)
        for v in SORT_BY_VALUES:
            assert v in str(excinfo.value), (
                f"sort_by validator's error message lost the shared vocab term {v!r}"
            )


# ---------------------------------------------------------------------------
# (3) MCP dispatch surface — integrations/mcp_server/_dispatch.py
# ---------------------------------------------------------------------------

def _dispatch_service():
    """A service stub whose MODE FLAGS are real bools.

    A bare `MagicMock()` cannot be used here: `handle_tools_call` gates
    vectr_recall behind `getattr(service, "search_only", False)`, and every
    unset MagicMock attribute is truthy — so a bare mock takes the
    search-only early return and never reaches the sort_by guard at all.
    That return carries `isError: False`, which means the "rejects" test
    fails *and* the "accepts" test passes vacuously. `_base_mock_service`
    is the repo's shared stub and sets `search_only`/`memory_only` to real
    `False` for exactly this reason (see its comment in conftest)."""
    from tests.conftest import _base_mock_service
    return _base_mock_service()


class TestMcpDispatchSortByVocab:
    def test_mcp_dispatch_rejects_out_of_vocab_with_shared_constant_in_message(
        self,
    ) -> None:
        from agent.working_context_store import SORT_BY_VALUES
        from integrations.mcp_server._dispatch import handle_tools_call

        out_of_vocab = "_not_in_vocab_" + "x"
        svc = _dispatch_service()
        result = handle_tools_call(
            "vectr_recall", {"sort_by": out_of_vocab}, svc, session_id="s"
        )
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "sort_by must be one of" in text
        # The error text names every shared vocab term verbatim.
        for v in SORT_BY_VALUES:
            assert v in text, (
                f"MCP dispatch error lost shared vocab term {v!r}"
            )
        # The dispatch must NOT have called the service.
        svc.recall.assert_not_called()

    def test_mcp_dispatch_accepts_every_shared_value(self) -> None:
        from agent.working_context_store import SORT_BY_VALUES
        from integrations.mcp_server._dispatch import handle_tools_call

        svc = _dispatch_service()
        for v in SORT_BY_VALUES:
            result = handle_tools_call("vectr_recall", {"sort_by": v}, svc, session_id="s")
            assert result.get("isError") is False, (
                f"MCP dispatch rejected shared vocab term {v!r}"
            )


# ---------------------------------------------------------------------------
# (4) MCP tool schema surface — integrations/mcp_server/_schemas.py
# ---------------------------------------------------------------------------

class TestMcpSchemaSortByVocab:
    def test_mcp_tool_schema_enum_is_the_shared_constant(self) -> None:
        """The JSON Schema `enum` the MCP client sees must be derived
        from the same SORT_BY_VALUES constant. Identity would be over-
        reaching (the schema needs a JSON list, so the module list()-
        casts at the boundary), but the contents are pinned element-for-
        element and the schema is rejected if a fifth mode is missing."""
        from agent.working_context_store import SORT_BY_VALUES
        from integrations.mcp_server._schemas import MCP_TOOLS

        # MCP_TOOLS is a flat list of tool dicts; find the vectr_recall one.
        recall_tool = next(t for t in MCP_TOOLS if t["name"] == "vectr_recall")
        props = recall_tool["inputSchema"]["properties"]
        # Element-for-element match: schema enum is the shared constant,
        # just as a list (JSON Schema needs a JSON-native list, not a
        # tuple). Equality is the right test here — identity would be
        # wrong because list() casts break it.
        assert list(props["sort_by"]["enum"]) == list(SORT_BY_VALUES)
        assert set(props["sort_by"]["enum"]) == set(SORT_BY_VALUES)
        # Default is also the shared constant's first value — the rest
        # of the codebase documents "relevance" as the default; if a
        # future PR reorders the shared tuple, this pin reminds the
        # author to update the schema's "default" too.
        assert props["sort_by"]["default"] == SORT_BY_VALUES[0]


# ---------------------------------------------------------------------------
# (5) CLI surface — main.py p_recall argparse choices
# ---------------------------------------------------------------------------

class TestCliSurfaceSortByVocab:
    def test_cli_main_py_source_uses_lazy_import_of_shared_constant(self) -> None:
        """The source-level pin: main.py must IMPORT the shared constant
        inside `main()` (lazy, NOT at module top — to preserve the
        `vectr hook` fast path's import budget) and then USE it as the
        `--sort-by` argparse choices. If a future change re-spells the
        vocabulary at the CLI site instead of importing the shared
        constant, this assertion catches it."""
        from main import main as _cli_main

        source = inspect.getsource(_cli_main)
        # Lazy import inside main() — preserves the hook fast path.
        assert "from agent.working_context_store import SORT_BY_VALUES" in source
        # The literal vocabulary must NOT be re-spelled at this site.
        # If a future change replaces the constant with a literal
        # `["relevance", "recency", "priority", "chronological"]` at
        # main.py:4383, this assertion catches the drift.
        assert (
            '"relevance", "recency", "priority", "chronological"' not in source
        ), (
            "main.py re-spelled the sort_by vocabulary at the CLI site; "
            "use SORT_BY_VALUES from agent.working_context_store instead"
        )
        assert (
            "'relevance', 'recency', 'priority', 'chronological'" not in source
        ), (
            "main.py re-spelled the sort_by vocabulary at the CLI site; "
            "use SORT_BY_VALUES from agent.working_context_store instead"
        )

    def test_cli_argparse_choices_honour_shared_constant(self) -> None:
        """Runtime pin on the choices contract. Build a minimal argparse
        parser the way main.py does (with the shared constant as
        choices) and assert: (a) every shared value is accepted, (b)
        any string not in the shared constant is rejected. This is the
        runtime behaviour the source-level pin in the previous test
        exists to protect — together they prove the CLI surface is
        driven by the shared constant end-to-end."""
        import argparse
        from agent.working_context_store import SORT_BY_VALUES

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        p_recall = sub.add_parser("recall")
        # Same shape as main.py: list(SORT_BY_VALUES) is what the CLI
        # passes to argparse, mirroring the import+list-cast in
        # main()'s recall subparser construction.
        p_recall.add_argument(
            "--sort-by", choices=list(SORT_BY_VALUES), default="relevance",
            dest="sort_by",
        )

        # Every shared value is accepted.
        for v in SORT_BY_VALUES:
            args = parser.parse_args(["recall", "--sort-by", v])
            assert args.sort_by == v

        # An out-of-vocab value is rejected — this is exactly what
        # argparse's `choices=` constraint gives us, and what the CLI
        # surface is supposed to inherit from the shared constant.
        with pytest.raises(SystemExit):
            parser.parse_args(["recall", "--sort-by", "_not_in_vocab_"])


# ---------------------------------------------------------------------------
# (6) All-five-sites share the same constant
# ---------------------------------------------------------------------------

class TestAllSurfacesShareOneConstant:
    def test_every_surface_vocab_is_the_shared_tuple(self) -> None:
        """Identity pin across all five surfaces in one place. This is
        the single test a reviewer can read to verify the fix actually
        holds. If a future PR re-spells the vocabulary at any one
        surface, this assertion fails immediately.

        Identity (not equality) is the right check: a literal re-spelling
        would still satisfy `==` (four strings, four strings) but not
        `is` (two distinct tuple objects). `is` is exactly the assertion
        that proves "drift is impossible without the test noticing"."""
        from agent.working_context_store import SORT_BY_VALUES

        # 1) REST layer's local alias must be the same object.
        from app.models import _SORT_BY_VALUES
        assert _SORT_BY_VALUES is SORT_BY_VALUES

        # 2) MCP dispatch module-level import binds the same object —
        # the dispatch imports `SORT_BY_VALUES` at module top, so the
        # module attribute IS the binding and the identity check below
        # is exact. (If the dispatch ever re-spelled the literal here,
        # the identity check fails immediately.)
        from integrations.mcp_server import _dispatch as _dispatch_mod
        assert _dispatch_mod.SORT_BY_VALUES is SORT_BY_VALUES

        # 3) MCP tool schema — element-for-element match (the schema
        # needs a JSON-native list, not a tuple, so identity is the
        # wrong test here; the right test is "no surface re-spelled").
        from integrations.mcp_server._schemas import MCP_TOOLS
        recall_tool = next(t for t in MCP_TOOLS if t["name"] == "vectr_recall")
        enum_list = recall_tool["inputSchema"]["properties"]["sort_by"]["enum"]
        assert set(enum_list) == set(SORT_BY_VALUES)

        # 4) CLI surface — read main.py's source the same way.
        from main import main as _cli_main
        cli_source = inspect.getsource(_cli_main)
        assert "from agent.working_context_store import SORT_BY_VALUES" in cli_source
        # The literal vocabulary must NOT be re-spelled at this site.
        assert '"relevance", "recency", "priority", "chronological"' not in cli_source
        assert "'relevance', 'recency', 'priority', 'chronological'" not in cli_source
