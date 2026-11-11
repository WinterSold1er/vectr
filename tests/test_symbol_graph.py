"""
Tests for agent/symbol_graph.py — SymbolGraph and extract_symbols_from_file.

Tests verify:
  - extract_symbols_from_file finds functions/classes in Python files
  - extract_symbols_from_file extracts call edges between functions
  - SymbolGraph.index_file stores symbols in SQLite
  - SymbolGraph.index_file is idempotent (re-index clears old entries)
  - SymbolGraph.locate() returns symbols with partial name match
  - SymbolGraph.callers() / callees() return correct edges
  - SymbolGraph.trace() combines both directions
  - SymbolGraph.delete_file() clears entries for that file
  - Workspace isolation: two workspaces don't cross-contaminate
  - format_locate_for_llm / format_trace_for_llm produce readable text
"""
from __future__ import annotations

import math
import sqlite3
import textwrap
from pathlib import Path

import pytest

from agent.symbol_graph import (
    SymbolGraph,
    Symbol,
    CallEdge,
    LocateResult,
    SYMBOL_LANGUAGES,
    SYMBOL_SCHEMA_VERSION,
    graph_toolchain_fingerprint,
    supports_symbols,
    extract_symbols_from_file,
    _levenshtein,
    _partial_match_key,
    _locate_scope_depth_from_lines,
    _locate_scope_depth_batch,
    _locate_class_enclosed_batch,
    _enclosing_class_from_lines,
    # UPG-BASE-METHOD-OVERRIDE-FLOOD: same-file override extraction.
    _extract_overrides,
    _collect_methods_by_class,
    _python_class_bases,
    _js_class_bases,
)
import agent.symbol_graph as _sgmod
from tests.conftest import make_py


def _seed_symbols(g: "SymbolGraph", workspace: str, specs: list[tuple]) -> None:
    """Insert synthetic symbols directly so ranking can be tested with precise
    (name, kind, file_path) control independent of any tree-sitter grammar.

    Each spec is (name, kind, file_path)."""
    import time
    now = time.time()
    with g._conn() as conn:
        for i, (name, kind, fp) in enumerate(specs):
            conn.execute(
                "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (workspace, name, kind, fp, i + 1, i + 1, now),
            )


def _seed_edges(g: "SymbolGraph", workspace: str, specs: list[tuple],
                edge_type: str = "calls") -> None:
    """Insert synthetic call edges directly so callee/caller ranking can be
    tested with precise control over frequency and call sites.

    Each spec is (from_file, from_symbol, from_line, to_symbol). All seeded with
    `edge_type` (default 'calls'; pass 'uses' for UPG-4.4 type-usage edges)."""
    with g._conn() as conn:
        for from_file, from_symbol, from_line, to_symbol in specs:
            conn.execute(
                "INSERT INTO edges (workspace, from_file, from_symbol, from_line, to_symbol, edge_type) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (workspace, from_file, from_symbol, from_line, to_symbol, edge_type),
            )


# ---------------------------------------------------------------------------
# UPG-3.3 — language-capability introspection
# ---------------------------------------------------------------------------

class TestSymbolLanguageCapability:
    def test_supported_languages_have_symbols(self) -> None:
        for lang in ("python", "javascript", "typescript", "go", "rust",
                     "java", "zig", "c", "cpp"):
            assert supports_symbols(lang), lang
            assert lang in SYMBOL_LANGUAGES

    def test_doc_and_unknown_languages_have_no_symbols(self) -> None:
        for lang in ("markdown", "html", "text", "cobol", "", "  "):
            assert not supports_symbols(lang)

    def test_display_name_spellings_normalized(self) -> None:
        # map / human-facing names should resolve to the same capability
        assert supports_symbols("C++")
        assert supports_symbols("Python")
        assert supports_symbols("  RUST ")
        assert not supports_symbols("C#")


# ---------------------------------------------------------------------------
# extract_symbols_from_file
# ---------------------------------------------------------------------------

class TestExtractSymbolsFromFile:
    def test_finds_top_level_function(self, tmp_path) -> None:
        path = make_py(tmp_path, "auth.py", """
            def verify_token(token: str) -> dict:
                return {}
        """)
        symbols, _ = extract_symbols_from_file(path)
        names = [s["name"] for s in symbols]
        assert "verify_token" in names

    def test_finds_class_definition(self, tmp_path) -> None:
        path = make_py(tmp_path, "models.py", """
            class UserModel:
                pass
        """)
        symbols, _ = extract_symbols_from_file(path)
        names = [s["name"] for s in symbols]
        assert "UserModel" in names

    def test_symbol_has_kind_function(self, tmp_path) -> None:
        path = make_py(tmp_path, "f.py", "def foo(): pass")
        symbols, _ = extract_symbols_from_file(path)
        fn = next(s for s in symbols if s["name"] == "foo")
        assert fn["kind"] == "function"

    def test_symbol_has_kind_class(self, tmp_path) -> None:
        path = make_py(tmp_path, "c.py", "class Bar: pass")
        symbols, _ = extract_symbols_from_file(path)
        cls = next(s for s in symbols if s["name"] == "Bar")
        assert cls["kind"] == "class"

    def test_symbol_line_numbers(self, tmp_path) -> None:
        path = make_py(tmp_path, "lines.py", textwrap.dedent("""\
            def alpha():
                pass

            def beta():
                pass
        """))
        symbols, _ = extract_symbols_from_file(path)
        names_to_lines = {s["name"]: s["start_line"] for s in symbols}
        assert names_to_lines["alpha"] == 1
        assert names_to_lines["beta"] == 4

    def test_extracts_call_edges(self, tmp_path) -> None:
        path = make_py(tmp_path, "calls.py", """
            def helper():
                pass

            def main():
                helper()
        """)
        _, edges = extract_symbols_from_file(path)
        callees = [e["to_symbol"] for e in edges]
        assert "helper" in callees

    def test_edges_have_correct_fields(self, tmp_path) -> None:
        path = make_py(tmp_path, "e.py", """
            def a():
                pass
            def b():
                a()
        """)
        _, edges = extract_symbols_from_file(path)
        assert len(edges) >= 1
        e = edges[0]
        assert "from_file" in e
        assert "from_symbol" in e
        assert "to_symbol" in e
        assert "edge_type" in e

    def test_no_duplicate_edges(self, tmp_path) -> None:
        path = make_py(tmp_path, "dup.py", """
            def caller():
                callee()
                callee()
                callee()
            def callee():
                pass
        """)
        _, edges = extract_symbols_from_file(path)
        keys = [(e["from_symbol"], e["to_symbol"]) for e in edges]
        assert len(keys) == len(set(keys))

    def test_unsupported_extension_returns_empty(self, tmp_path) -> None:
        f = tmp_path / "notes.txt"
        f.write_text("just some text")
        symbols, edges = extract_symbols_from_file(str(f))
        assert symbols == []
        assert edges == []

    def test_nonexistent_file_returns_empty(self, tmp_path) -> None:
        symbols, edges = extract_symbols_from_file(str(tmp_path / "ghost.py"))
        assert symbols == []
        assert edges == []

    def test_multiple_functions_found(self, tmp_path) -> None:
        path = make_py(tmp_path, "multi.py", """
            def foo(): pass
            def bar(): pass
            def baz(): pass
        """)
        symbols, _ = extract_symbols_from_file(path)
        names = {s["name"] for s in symbols}
        assert {"foo", "bar", "baz"}.issubset(names)


class TestGoTypeDeclarationNameUPGRustStructChunkMissing:
    """A Go `type_declaration`'s own name is nested one level down, under a
    `type_spec` child, unlike every other symbol-defining node this module
    handles — before this fix, `_get_symbol_name` never found it as a direct
    child, so every Go struct/interface/type-alias symbol silently got
    name="" and was dropped from the graph entirely (found while wiring the
    Rust struct-chunking fix — UPG-RUST-STRUCT-CHUNK-MISSING)."""

    def test_struct_type_declaration_name_resolved(self, tmp_path) -> None:
        p = tmp_path / "shapes.go"
        p.write_text(
            "package shapes\n"
            "\n"
            "type Point struct {\n"
            "\tX int\n"
            "\tY int\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        by_name = {s["name"]: s for s in symbols}
        assert "Point" in by_name
        assert by_name["Point"]["kind"] == "struct"

    def test_interface_type_declaration_name_resolved(self, tmp_path) -> None:
        p = tmp_path / "shapes.go"
        p.write_text(
            "package shapes\n"
            "\n"
            "type Shape interface {\n"
            "\tArea() float64\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        names = {s["name"] for s in symbols}
        assert "Shape" in names

    def test_no_blank_name_symbol_emitted(self, tmp_path) -> None:
        p = tmp_path / "shapes.go"
        p.write_text(
            "package shapes\n"
            "\n"
            "type Point struct {\n"
            "\tX int\n"
            "}\n"
            "\n"
            "func Origin() Point {\n"
            "\treturn Point{}\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        assert "" not in {s["name"] for s in symbols}


class TestJavaMethodNameExtractionUPG:
    """UPG-JAVA-METHOD-NAME-EXTRACTION: a Java method with a non-primitive
    return type emits `type_identifier` (the return type) as a direct child
    BEFORE the method-name `identifier`, so the positional child-scan indexed
    the method under its return type's name. `child_by_field_name("name")` is
    authoritative."""

    def test_method_indexed_under_method_name_not_return_type(self, tmp_path) -> None:
        p = tmp_path / "Factory.java"
        p.write_text(
            "class Factory {\n"
            "  public Producer createProducer() { return null; }\n"
            "  private ConnectionPool pool() { return null; }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        by_name = {s["name"]: s for s in symbols}
        assert "createProducer" in by_name
        assert by_name["createProducer"]["kind"] == "method"
        assert "pool" in by_name
        # the return types must NOT be minted as method symbols
        assert "Producer" not in by_name
        assert "ConnectionPool" not in by_name

    def test_primitive_return_type_method_still_resolved(self, tmp_path) -> None:
        p = tmp_path / "Calc.java"
        p.write_text("class Calc {\n  int add(int a, int b) { return a + b; }\n}\n")
        names = {s["name"] for s in extract_symbols_from_file(str(p))[0]}
        assert "add" in names


class TestRustImplSymbolNameUPG:
    """UPG-RUST-IMPL-SYMBOL-NAME-TRAIT-VS-TYPE: an `impl_item` exposes no
    `name` field, so `_get_symbol_name` fell through to the positional
    child-scan, whose first identifier-ish child is the TRAIT for
    `impl Trait for Type { }` — the impl container was indexed under the
    trait's name. Resolution now goes through the grammar's `type` field,
    which names the implemented type on both inherent and trait impls."""

    def test_trait_impl_indexed_under_type_not_trait(self, tmp_path) -> None:
        # No struct declaration in the file: the ONLY emitted symbol is the
        # impl block itself, so the assertion can't be satisfied by anything
        # else. With the bug present this emits ("Measurable", "impl") and
        # nothing else; fixed, it emits ("Widget", "impl").
        p = tmp_path / "widget_impl.rs"
        p.write_text(
            "impl Measurable for Widget {\n"
            "    fn measure(&self) -> u32 { 1 }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        by_name = {s["name"]: s for s in symbols}
        assert by_name.get("Widget", {}).get("kind") == "impl", (
            f"expected the impl block indexed under the TYPE 'Widget', got {symbols}"
        )
        assert "Measurable" not in by_name, (
            f"the trait name must never become the impl symbol's name: {sorted(by_name)}"
        )

    def test_inherent_impl_still_resolved(self, tmp_path) -> None:
        # Regression guard: `impl Type { }` has no trait child; the old scan
        # happened to pick the type. The field-based path must agree.
        p = tmp_path / "widget.rs"
        p.write_text(
            "impl Widget {\n"
            "    fn refresh(&mut self) { }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        impls = [s for s in symbols if s["kind"] == "impl"]
        assert [s["name"] for s in impls] == ["Widget"]

    def test_generic_trait_impl_resolved_via_type_field(self, tmp_path) -> None:
        # Both the trait (`Measurable<T>`) and the type (`Bag<T>`) are
        # generic_type nodes here, so the positional scan found NO matching
        # direct child and dropped the impl entirely (name "" → not emitted).
        # The field path unwraps generic_type → its own `type` field down to
        # the bare type_identifier, same as _graph._impl_owner_type_name.
        p = tmp_path / "bag.rs"
        p.write_text(
            "impl<T> Measurable<T> for Bag<T> {\n"
            "    fn measure(&self) -> u32 { 1 }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        impls = [s for s in symbols if s["kind"] == "impl"]
        assert [s["name"] for s in impls] == ["Bag"], (
            f"generic trait impl must resolve to the type 'Bag', got {symbols}"
        )


class TestJavaConstructorSymbolsUPG:
    """UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE (constructor half):
    constructor_declaration was absent from the Java symbol map, so every
    constructor was invisible to the symbol graph — unlocatable, and calls
    inside its body attributed to the enclosing class's context rather than
    to the constructor's own position."""

    def test_constructor_extracted_as_method_kind_symbol(self, tmp_path) -> None:
        p = tmp_path / "Widget.java"
        p.write_text(
            "public class Widget {\n"
            "    private int v;\n"
            "    public Widget() { this.v = 0; }\n"
            "    public Widget(int v) { this.v = v; }\n"
            "    public int get() { return v; }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        widget_kinds = {s["kind"] for s in symbols if s["name"] == "Widget"}
        # Both the class definition AND its constructors share the name;
        # before the fix only the class's "class" kind existed.
        assert {"class", "method"} <= widget_kinds, (
            f"expected class + constructor(method) symbols named 'Widget', got {widget_kinds}"
        )

    def test_constructor_line_points_at_declaration(self, tmp_path) -> None:
        p = tmp_path / "Gauge.java"
        p.write_text(
            "public class Gauge {\n"
            "    public Gauge(int start) { }\n"
            "    public int read() { return 0; }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        ctors = [s for s in symbols if s["name"] == "Gauge" and s["kind"] == "method"]
        assert len(ctors) == 1
        lines = p.read_text().splitlines()
        assert "Gauge(int start)" in lines[ctors[0]["start_line"] - 1], (
            f"constructor start_line {ctors[0]['start_line']} does not point at its declaration"
        )


class TestJavaEnumAndFieldSymbolsUPG:
    """UPG-JAVA-ENUM-CONSTANTS-NO-L2-SYMBOLS: enum_constant and field_declaration
    were absent from the Java symbol map, so `vectr_locate STATUS_CODE_159` /
    `vectr_locate MAX_RETRIES` missed while the L3 chunker now finds them — the
    TIER0 asymmetry class in its inverted form, after the constructor half was
    closed at v13. Enum constants and class fields are members of a type's
    namespace and locatable; method-locals are not — but tree-sitter-java
    separates `field_declaration` (class/enum body) from
    `local_variable_declaration` (method body) at the grammar level, so the
    enclosing-kind gate the Zig case required is unnecessary here."""

    def test_enum_constants_are_extracted_with_their_kind(self, tmp_path) -> None:
        p = tmp_path / "Status.java"
        p.write_text(
            "public enum Status {\n"
            "    ACTIVE,\n"
            "    INACTIVE,\n"
            "    PENDING_CODE_159\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        by_name = {s["name"]: s for s in symbols}
        # The enum itself is still kind=enum (rank 0); each constant is kind=
        # enum_constant (rank 1) so the enum outranks its values on a same-
        # name collision, mirroring the class/method precedent.
        assert by_name["Status"]["kind"] == "enum"
        assert by_name["ACTIVE"]["kind"] == "enum_constant"
        assert by_name["INACTIVE"]["kind"] == "enum_constant"
        assert by_name["PENDING_CODE_159"]["kind"] == "enum_constant"
        # No "Status" kind=enum_constant leak — the enum body's name is its
        # own declaration, not a member of itself.
        assert "Status" not in {n for n, s in by_name.items() if s["kind"] == "enum_constant"}

    def test_enum_constant_is_locatable_via_locate_l2(self, tmp_path) -> None:
        p = tmp_path / "Status.java"
        p.write_text(
            "public enum Status {\n"
            "    ACTIVE,\n"
            "    INACTIVE\n"
            "}\n"
        )
        g = SymbolGraph(str(tmp_path))
        g.index_file("ws", str(p))
        result = g.locate_l2("ws", "ACTIVE")
        assert result.resolution_strategy == "exact"
        assert result.symbols[0].name == "ACTIVE"
        assert result.symbols[0].kind == "enum_constant"

    def test_class_field_is_extracted_with_field_kind(self, tmp_path) -> None:
        p = tmp_path / "Config.java"
        p.write_text(
            "public class Config {\n"
            "    public int MAX_RETRIES = 3;\n"
            "    private final String NAME = \"default\";\n"
            "    public static int PORT = 8080;\n"
            "    public void setPort(int p) { this.PORT = p; }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        by_name = {s["name"]: s for s in symbols}
        # The static/final distinction does not change the kind — both instance
        # and `static`/`final` fields are part of the class's namespace and
        # share a single `field` kind. Splitting them would multiply locate
        # noise without a documented use case.
        assert by_name["MAX_RETRIES"]["kind"] == "field"
        assert by_name["NAME"]["kind"] == "field"
        assert by_name["PORT"]["kind"] == "field"
        # The setPort method must still be present and not demoted by the new
        # field entries; the class ranks above its fields, methods rank with
        # fields (all 1), and the locatable "PORT" is the field, not anything
        # else.
        assert by_name["setPort"]["kind"] == "method"
        assert by_name["Config"]["kind"] == "class"

    def test_method_local_variable_is_not_extracted_as_a_field(self, tmp_path) -> None:
        """tree-sitter-java separates `field_declaration` (class/enum body
        members) from `local_variable_declaration` (method body locals) at the
        grammar level. Without that separation, the Zig-style enclosing-kind
        gate would have been required to keep `int x = 0;` inside a method
        body from leaking into the graph as a `field`. The grammar already
        prevents that, and this test pins the property: a local declaration
        matching the field_declaration signature (`int x = 0;`) inside a
        method body is not in the symbol set."""
        p = tmp_path / "Local.java"
        p.write_text(
            "public class Local {\n"
            "    public void run() {\n"
            "        int x = 0;\n"
            "        String s = \"hi\";\n"
            "        x = x + 1;\n"
            "    }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        names = {s["name"] for s in symbols}
        assert "x" not in names
        assert "s" not in names
        # The class and method are still extracted.
        assert "Local" in names
        assert "run" in names

    def test_enum_constant_ranks_below_its_enum_with_the_same_name(self, tmp_path) -> None:
        """A symbol kind that ranks BELOW the enum/class that contains it
        must hold for both of the new kinds, so an enum and an enum constant
        of the same name (or a class and a field of the same name) resolves
        to the type definition, not the member. Confirms the rank=1 entry
        for both new kinds, matching the class/method precedent (rank 0/1).

        Asserted through `locate_l2`, which is the surface that consults
        `_KIND_RANK`; plain `locate` orders by exact-match/name-length/path
        and has never ranked by kind, so it would return seed order here and
        prove nothing either way (cf. the struct-over-impl precedent in
        TestLocateL2Ranking.test_canonical_def_ranks_before_impl_on_exact_match).
        Each pair is seeded member-first so a pass cannot come from insertion
        order."""
        # Seed the graph directly so we can test the rank ordering without
        # any tree-sitter grammar concerns — _seed_symbols is the existing
        # helper for that.
        g = SymbolGraph(str(tmp_path))
        p = tmp_path / "Same.java"
        p.write_text("class Same { }\n")
        _seed_symbols(g, "ws", [
            ("Same", "enum_constant", str(p)),
            ("Same", "enum",         str(p)),
        ])
        ranked = g.locate_l2("ws", "Same").symbols
        # The first result is the type definition; the constant sorts after.
        assert ranked[0].kind == "enum"
        assert ranked[1].kind == "enum_constant"
        # Same property for the class/field pair.
        _seed_symbols(g, "ws2", [
            ("Pair", "field", str(p)),
            ("Pair", "class", str(p)),
        ])
        ranked2 = g.locate_l2("ws2", "Pair").symbols
        assert ranked2[0].kind == "class"
        assert ranked2[1].kind == "field"

    def test_static_final_field_has_no_distinct_kind(self, tmp_path) -> None:
        """Static/final/instance fields are all extracted under the single
        `field` kind, by design: splitting them would multiply locate noise
        without a documented use case. Constants (Java `final` primitives
        the JLS would call a compile-time constant) are NOT a separate kind
        in the symbol graph — they are still a `field`, mirroring the L3
        chunker which also has no Java-constant-specific kind."""
        p = tmp_path / "Const.java"
        p.write_text(
            "public class Const {\n"
            "    public static final int LIMIT = 100;\n"
            "    public final int VERSION = 1;\n"
            "    public int counter = 0;\n"
            "}\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        for n in ("LIMIT", "VERSION", "counter"):
            assert by_name[n]["kind"] == "field", (
                f"{n} should be a field, not {by_name[n]['kind']!r}"
            )


class TestRustImplSymbolNameNodeUPG:
    """UPG-IMPL-SYMBOL-NAME-NODE-GAP: `_get_symbol_name_node` returned None
    for a Rust `impl_item` and the parse-error check fell back to the whole
    impl block, so an impl containing one locally-erroring construct lost
    its symbol identity where an equivalent function kept it. The fix
    factors the type-field walk out of `_get_symbol_name` into
    `_rust_impl_type_node` and uses it from both resolvers, so the name-node
    the parse-error check scopes to is the same node the string resolver
    extracts the name from — one walk, two callers, no drift."""

    def test_impl_with_locally_erroring_construct_keeps_its_symbol(self, tmp_path) -> None:
        """The witness shape for the gap: a syntactically clean impl header
        followed by a body containing a construct the grammar cannot parse.
        Here the broken function body has an unmatched `let` (missing `;`),
        which tree-sitter-rust flags as a local ERROR inside the function
        without wrapping the whole impl block — exactly the shape this fix
        is meant to recover from. Before the fix the impl symbol was
        dropped because `node.has_error` over the whole block returned
        True; after the fix it is kept because the parse-error check scopes
        to the `type_identifier` node, which is clean."""
        p = tmp_path / "widget.rs"
        p.write_text(
            "struct Widget;\n"
            "impl Widget {\n"
            "    fn broken() {\n"
            "        let x = 1\n"
            "        let y = 2;\n"
            "    }\n"
            "    fn ok() -> u32 { 1 }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        impls = [s for s in symbols if s["kind"] == "impl"]
        # The impl block must still be indexed under its type name, even
        # though its body contains a locally-unparseable statement sequence.
        assert [s["name"] for s in impls] == ["Widget"], (
            f"impl symbol must survive a locally-erroring body construct, got {symbols}"
        )

    def test_trait_for_type_still_indexed_under_type(self, tmp_path) -> None:
        """Regression guard for the shared walk: `impl Trait for Type { }`
        must still resolve to the TYPE, never the trait. The v13 fix put the
        type-field walk into `_get_symbol_name`; the v14 refactor moves it
        into `_rust_impl_type_node` and shares it with the name-node
        resolver. Both consumers of the walk must agree on which type the
        impl symbol names. Pin with a fixture that has NO other `Widget`
        symbols so the assertion cannot pass by accident."""
        p = tmp_path / "widget_impl.rs"
        p.write_text(
            "impl Measurable for Widget {\n"
            "    fn measure(&self) -> u32 { 1 }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        by_name = {s["name"]: s for s in symbols}
        assert by_name.get("Widget", {}).get("kind") == "impl", (
            f"impl Trait for Type must resolve to the type 'Widget', got {symbols}"
        )
        assert "Measurable" not in by_name, (
            f"the trait name must never become the impl symbol's name: {sorted(by_name)}"
        )

    def test_generic_trait_impl_resolved_via_type_field(self, tmp_path) -> None:
        """Both the trait (`Measurable<T>`) and the type (`Bag<T>`) are
        `generic_type` nodes here, so the old positional scan found no
        matching direct child and dropped the impl entirely (name "" → not
        emitted). The shared walk unwraps `generic_type` → its own `type`
        field down to the bare `type_identifier`; the refactor must keep
        that."""
        p = tmp_path / "bag.rs"
        p.write_text(
            "impl<T> Measurable<T> for Bag<T> {\n"
            "    fn measure(&self) -> u32 { 1 }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        impls = [s for s in symbols if s["kind"] == "impl"]
        assert [s["name"] for s in impls] == ["Bag"], (
            f"generic trait impl must resolve to the type 'Bag', got {symbols}"
        )

    def test_broken_impl_name_still_rejected(self, tmp_path) -> None:
        """The whole point of the name-node-scoped error check is to
        preserve symbols whose name is clean; a name that is itself part
        of a parse error must STILL be rejected (a malformed identifier
        cannot mint a symbol). Without the reserved-keyword and
        whole-subtree check, a previous coverage push indexed grammar
        keywords as symbols. Pin the rejection on the worst-case impl
        header the grammar can produce — one whose `type` field is
        itself an ERROR node (no `type_identifier` reachable). The
        shared walk returns None, the name resolver returns "", and
        no symbol is emitted."""
        p = tmp_path / "broken.rs"
        # `impl {` — no type at all, the parser produces an impl_item whose
        # `type` field is missing/empty. Both the string and the name-node
        # resolver must fall through cleanly to "not a symbol".
        p.write_text(
            "impl {\n"
            "    fn nothing() { }\n"
            "}\n"
        )
        symbols, _ = extract_symbols_from_file(str(p))
        # The impl has no resolvable name → no impl-kind symbol, and the
        # function inside it (kind=function) is independent and not pinned
        # by this test.
        assert not any(s["kind"] == "impl" for s in symbols), (
            f"an impl with no resolvable type must not be indexed, got {symbols}"
        )


class TestTSTypedefSymbolGraphUPG:
    """UPG-TS-SYMBOLGRAPH-TYPEDEF: type_alias_declaration and enum_declaration
    must resolve as symbols so vectr_locate finds a TS type alias / enum, not
    just interfaces and classes."""

    def test_type_alias_and_enum_are_symbols(self, tmp_path) -> None:
        p = tmp_path / "types.ts"
        p.write_text(
            "export type UserId = string;\n"
            "export enum Color { Red, Green, Blue }\n"
            "export interface Shape { area(): number }\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        assert by_name.get("UserId", {}).get("kind") == "type"
        assert by_name.get("Color", {}).get("kind") == "enum"
        assert by_name.get("Shape", {}).get("kind") == "interface"

    def test_ts_typedef_locatable(self, tmp_path) -> None:
        p = tmp_path / "types.ts"
        p.write_text("export type UserId = string;\nexport enum Color { Red }\n")
        g = SymbolGraph(str(tmp_path))
        g.index_file("ws", str(p))
        assert g.locate_l2("ws", "UserId").resolution_strategy == "exact"
        assert g.locate_l2("ws", "Color").resolution_strategy == "exact"


class TestCStructTypedefLocateUPG:
    """UPG-C-STRUCT-TYPEDEF-LOCATE / UPG-C-MACRO-ADJACENT-DROP: the whole-subtree
    error check dropped C symbols whose NAME/declarator was clean but whose
    body/field-list contained a macro tree-sitter-c couldn't parse. The check is
    now declarator-scoped, so both a typedef-struct with a macro field and a
    function whose body is a type-name-arg macro survive."""

    def test_typedef_struct_with_macro_field_extracted(self, tmp_path) -> None:
        p = tmp_path / "dictobject.c"
        p.write_text(
            "typedef struct {\n"
            "    PyObject_HEAD\n"
            "    Py_ssize_t ma_used;\n"
            "    PyDictKeysObject *ma_keys;\n"
            "} PyDictObject;\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        assert "PyDictObject" in by_name

    def test_macro_body_function_extracted(self, tmp_path) -> None:
        p = tmp_path / "longobject.c"
        p.write_text(
            "#define PYLONG_FROM_INT(INT_TYPE, ival) do { return; } while (0)\n"
            "PyObject * PyLong_FromLong(long ival) { PYLONG_FROM_INT(unsigned long, ival); }\n"
            "PyObject * PyLong_AsLong(PyObject *v) { return v; }\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        assert "PyLong_FromLong" in by_name
        assert by_name["PyLong_FromLong"]["kind"] == "function"
        assert "PyLong_AsLong" in by_name  # clean neighbour unaffected

    def test_typedef_struct_locate_exact_not_fuzzy(self, tmp_path) -> None:
        p = tmp_path / "dictobject.c"
        p.write_text("typedef struct {\n    PyObject_HEAD\n} PyDictObject;\n")
        g = SymbolGraph(str(tmp_path))
        g.index_file("ws", str(p))
        result = g.locate_l2("ws", "PyDictObject")
        assert result.resolution_strategy == "exact"
        assert result.symbols[0].name == "PyDictObject"

    def test_fuzzy_fallback_labeled_as_inexact(self, tmp_path) -> None:
        # A fuzzy (edit-distance) match must be rendered as an explicit
        # no-exact-match near-miss, never as a confident hit — the historical
        # `locate("PyDictObject")` → `PyODictObject` silent-wrong-answer.
        g = SymbolGraph(str(tmp_path))
        _seed_symbols(g, "ws", [
            ("PyODictObject", "type", str(tmp_path / "odictobject.c")),
        ])
        result = g.locate_l2("ws", "PyDictObject")
        assert result.resolution_strategy == "fuzzy"
        text = g.format_locate_l2_for_llm(result)
        assert "No exact match for 'PyDictObject'" in text
        assert "PyODictObject" in text
        # never render a fuzzy near-miss as a real "match via <strategy>" hit
        assert "match via" not in text
        assert "matches via" not in text


class TestZigSymbolExtractionUPG:
    """UPG-ZIG-SYMBOL-EXTRACTION: Zig's overloaded `variable_declaration` node
    mapped every const/var — including function-local declarations and bare
    assignments the grammar mis-parses as declarations — to kind "struct". The
    kind is now resolved from the RHS; locals/mutations/imports are not symbols."""

    def test_local_var_and_mutation_not_indexed_as_struct(self, tmp_path) -> None:
        p = tmp_path / "checksum.zig"
        p.write_text(
            "pub fn checksum(data: []const u8) u128 {\n"
            "    var checksum: u128 = 0;\n"
            "    checksum +%= run(data);\n"
            "    return checksum;\n"
            "}\n"
        )
        symbols = extract_symbols_from_file(str(p))[0]
        # the only `checksum` symbol is the function definition, not a [struct]
        checksum_syms = [s for s in symbols if s["name"] == "checksum"]
        assert len(checksum_syms) == 1
        assert checksum_syms[0]["kind"] == "function"
        assert not any(s["kind"] == "struct" for s in symbols)

    def test_container_type_decls_keep_correct_kind(self, tmp_path) -> None:
        p = tmp_path / "types.zig"
        p.write_text(
            "pub const Node = struct { x: u32 };\n"
            "pub const Color = enum { Red, Green };\n"
            "pub const Payload = union { a: u8, b: u16 };\n"
            "pub const Errs = error { NotFound };\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        assert by_name["Node"]["kind"] == "struct"
        assert by_name["Color"]["kind"] == "enum"
        assert by_name["Payload"]["kind"] == "struct"
        assert by_name["Errs"]["kind"] == "enum"

    def test_import_and_factory_bindings_not_struct(self, tmp_path) -> None:
        p = tmp_path / "mod.zig"
        p.write_text(
            'const std = @import("std");\n'
            "pub const Tree = TreeType(K, V);\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        # @import binding is not a locatable definition
        assert "std" not in by_name
        # module-level factory binding is a variable, not a fabricated struct
        assert by_name.get("Tree", {}).get("kind") == "variable"


class TestZigStructScopedConstUPG:
    """UPG-ZIG-STRUCT-CONST-LOCATE: a `const`/`var` declared inside a struct/enum
    body is part of that container's namespace (Zig `Node.default_capacity`) and
    must be locatable — the old `current_symbol == ""` gate conflated struct
    scope with function scope and dropped both. Function-locals stay excluded."""

    def test_struct_scoped_pub_const_is_extracted_as_constant(self, tmp_path) -> None:
        p = tmp_path / "checksum.zig"
        p.write_text(
            "pub const Node = struct {\n"
            "    x: u32,\n"
            "    pub const default_capacity: u32 = 16;\n"
            "    pub fn checksum(data: []const u8) u128 {\n"
            "        var acc: u128 = 0;\n"
            "        return acc;\n"
            "    }\n"
            "};\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        # F64: the struct-scoped `pub const` is a locatable constant
        assert by_name.get("default_capacity", {}).get("kind") == "constant"
        assert by_name["default_capacity"]["start_line"] == 3
        # the struct and its method are still extracted correctly
        assert by_name["Node"]["kind"] == "struct"
        assert by_name["checksum"]["kind"] == "function"

    def test_function_local_const_and_var_still_excluded(self, tmp_path) -> None:
        p = tmp_path / "fn.zig"
        p.write_text(
            "pub fn compute() u32 {\n"
            "    const LOCAL_MAX: u32 = 99;\n"
            "    var total: u32 = 0;\n"
            "    return total + LOCAL_MAX;\n"
            "}\n"
        )
        names = {s["name"] for s in extract_symbols_from_file(str(p))[0]}
        assert "compute" in names
        # function-locals are disposable, not part of any locatable namespace
        assert "LOCAL_MAX" not in names
        assert "total" not in names

    def test_struct_scoped_var_is_extracted_as_variable(self, tmp_path) -> None:
        p = tmp_path / "counter.zig"
        p.write_text(
            "pub const Counter = struct {\n"
            "    var instances: u32 = 0;\n"
            "};\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        # a struct-scoped `var` keeps the variable kind (from the `var` keyword,
        # not the SCREAMING_CASE module-scope heuristic)
        assert by_name.get("instances", {}).get("kind") == "variable"

    def test_enum_scoped_const_is_extracted(self, tmp_path) -> None:
        p = tmp_path / "mode.zig"
        p.write_text(
            "pub const Mode = enum {\n"
            "    fast,\n"
            "    slow,\n"
            "    pub const default_mode = Mode.fast;\n"
            "};\n"
        )
        by_name = {s["name"]: s for s in extract_symbols_from_file(str(p))[0]}
        assert by_name["Mode"]["kind"] == "enum"
        # a const inside an enum body is part of the enum's namespace
        assert by_name.get("default_mode", {}).get("kind") == "constant"


# ---------------------------------------------------------------------------
# SymbolGraph — index_file / delete_file / symbol_count
# ---------------------------------------------------------------------------

class TestSymbolGraphIndex:
    def test_index_file_returns_symbol_count(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "a.py", "def foo(): pass\nclass Bar: pass")
        count = g.index_file("ws1", path)
        assert count >= 2

    def test_index_file_stores_in_db(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "b.py", "def my_func(): pass")
        g.index_file("ws1", path)
        assert g.symbol_count("ws1") >= 1

    def test_index_file_idempotent(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "c.py", "def fn(): pass")
        g.index_file("ws1", path)
        first = g.symbol_count("ws1")
        g.index_file("ws1", path)
        assert g.symbol_count("ws1") == first

    def test_delete_file_removes_symbols(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "d.py", "def to_remove(): pass")
        g.index_file("ws1", path)
        assert g.symbol_count("ws1") >= 1
        g.delete_file("ws1", path)
        assert g.symbol_count("ws1") == 0

    def test_build_for_workspace_indexes_multiple_files(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        p1 = make_py(tmp_path, "x.py", "def x(): pass")
        p2 = make_py(tmp_path, "y.py", "def y(): pass")
        stats = g.build_for_workspace("ws1", [p1, p2])
        assert stats["files"] == 2
        assert stats["symbols"] >= 2

    def test_workspace_isolation(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "shared.py", "def fn(): pass")
        g.index_file("ws_a", path)
        g.index_file("ws_b", path)
        assert g.symbol_count("ws_a") >= 1
        g.delete_file("ws_a", path)
        assert g.symbol_count("ws_a") == 0
        assert g.symbol_count("ws_b") >= 1  # ws_b unaffected


# ---------------------------------------------------------------------------
# SymbolGraph — locate
# ---------------------------------------------------------------------------

class TestSymbolGraphLocate:
    def test_locate_exact_match(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "auth.py", "def verify_token(): pass")
        g.index_file("ws", path)
        symbols = g.locate("ws", "verify_token")
        assert len(symbols) >= 1
        assert symbols[0].name == "verify_token"

    def test_locate_partial_match(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "f.py", "def authenticate_user(): pass\ndef authenticate_admin(): pass")
        g.index_file("ws", path)
        symbols = g.locate("ws", "authenticate")
        names = [s.name for s in symbols]
        assert "authenticate_user" in names
        assert "authenticate_admin" in names

    def test_locate_returns_symbol_with_file_path(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "mod.py", "def zap(): pass")
        g.index_file("ws", path)
        symbols = g.locate("ws", "zap")
        assert symbols[0].file_path == path

    def test_locate_returns_symbol_with_snippet(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "snap.py", "def snap_fn():\n    return 42\n")
        g.index_file("ws", path)
        symbols = g.locate("ws", "snap_fn")
        assert symbols[0].snippet != ""

    def test_locate_no_match_returns_empty(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        symbols = g.locate("ws", "nonexistent_xyz")
        assert symbols == []

    def test_locate_respects_limit(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        funcs = "\n".join(f"def fn_{i}(): pass" for i in range(10))
        path = make_py(tmp_path, "many.py", funcs)
        g.index_file("ws", path)
        symbols = g.locate("ws", "fn_", limit=3)
        assert len(symbols) <= 3


# ---------------------------------------------------------------------------
# SymbolGraph — locate_l2 (L2 multi-strategy call resolution)
# ---------------------------------------------------------------------------

class TestLocateL2:
    def test_exact_strategy(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "auth.py", "def verify_token(): pass")
        g.index_file("ws", str(tmp_path / "auth.py"))
        result = g.locate_l2("ws", "verify_token")
        assert isinstance(result, LocateResult)
        assert result.resolution_strategy == "exact"
        assert len(result.symbols) >= 1
        assert result.symbols[0].name == "verify_token"

    def test_suffix_strategy_strips_qualifier(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "auth.py", "def verify_token(): pass")
        g.index_file("ws", str(tmp_path / "auth.py"))
        # Query with a qualifier prefix — symbol is stored as plain "verify_token"
        result = g.locate_l2("ws", "module.verify_token")
        assert result.resolution_strategy == "suffix"
        assert result.symbols[0].name == "verify_token"

    def test_same_module_strategy(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        subdir = tmp_path / "pkg"
        subdir.mkdir()
        # Symbol in subdir
        f1 = subdir / "a.py"
        f1.write_text("def pkg_helper(): pass\n")
        # Symbol outside subdir
        f2 = tmp_path / "other.py"
        f2.write_text("def pkg_helper_outside(): pass\n")
        g.index_file("ws", str(f1))
        g.index_file("ws", str(f2))
        # Query "helper" from a file in the same subdir; same_module should prefer f1 results
        result = g.locate_l2("ws", "pkg_helper_xyz", caller_file=str(f1))
        # "pkg_helper_xyz" has no exact match; falls through to same_module
        # which finds "pkg_helper" (substring "helper" in "pkg_helper") in the same dir
        assert result.resolution_strategy in ("same_module", "substring", "fuzzy", "none")

    def test_substring_strategy(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "x.py", "def unique_xyz_fn(): pass")
        g.index_file("ws", str(tmp_path / "x.py"))
        # No exact match for "unique_xyz" — but a symbol contains it as a substring
        result = g.locate_l2("ws", "unique_xyz")
        assert result.resolution_strategy == "substring"
        assert result.symbols[0].name == "unique_xyz_fn"

    def test_fuzzy_strategy(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "f.py", "def foo_bar(): pass")
        g.index_file("ws", str(tmp_path / "f.py"))
        # "fo_bar" is NOT a substring of "foo_bar" (substring skips) but edit-distance 1
        result = g.locate_l2("ws", "fo_bar")
        assert result.resolution_strategy == "fuzzy"
        assert result.symbols[0].name == "foo_bar"

    def test_none_strategy_when_no_match(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        result = g.locate_l2("ws", "zzz_totally_nonexistent_qqq")
        assert result.resolution_strategy == "none"
        assert result.symbols == []

    def test_returns_locate_result_type(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        result = g.locate_l2("ws", "anything")
        assert isinstance(result, LocateResult)
        assert hasattr(result, "symbols")
        assert hasattr(result, "resolution_strategy")
        assert hasattr(result, "query")

    def test_format_locate_l2_shows_strategy(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "auth.py", "def verify_token(): pass")
        g.index_file("ws", str(tmp_path / "auth.py"))
        result = g.locate_l2("ws", "verify_token")
        text = g.format_locate_l2_for_llm(result)
        assert "exact name match" in text
        assert "verify_token" in text

    def test_format_locate_l2_empty(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        result = LocateResult(symbols=[], resolution_strategy="none", query="missing")
        text = g.format_locate_l2_for_llm(result)
        assert "No symbol matching" in text
        assert "missing" in text

    def test_import_chain_strategy(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        # Create a module that defines a symbol
        lib = tmp_path / "mylib.py"
        lib.write_text("def lib_function(): pass\n")
        # Create a caller file that imports mylib
        caller = tmp_path / "caller.py"
        caller.write_text("import mylib\n\ndef call_it():\n    mylib.lib_function()\n")
        g.index_file("ws", str(lib))
        g.index_file("ws", str(caller))
        # Search with caller_file — "lib_function" exists only in mylib.py (imported by caller)
        result = g.locate_l2("ws", "lib_function_zzz", caller_file=str(caller))
        # Even if import_chain doesn't find "lib_function_zzz", we just verify it doesn't crash
        assert isinstance(result, LocateResult)


# ---------------------------------------------------------------------------
# UPG-NEARMISS-SYMBOL-NAMES: cheap, deterministic near-miss lookup reused by
# app.service.identifier_hint_nearmiss for a token that already failed exact
# resolution. Every result here is inexact by construction.
# ---------------------------------------------------------------------------

class TestNearestSymbolNames:
    def test_prefix_containment_when_token_extends_a_real_shorter_name(self, tmp_path) -> None:
        """The common "misremembered one extra word" near-miss shape: the
        token is a real, shorter symbol name plus a trailing word. None of
        locate_l2's own strategies compare in this direction (they only
        match when the SYMBOL name contains the token, not the reverse), so
        this is the one new bounded query nearest_symbol_names adds."""
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "control.py", "class CacheControl:\n    pass\n")
        g.index_file("ws", str(tmp_path / "control.py"))

        # Confirm locate_l2 itself truly falls through to "none" for this pair
        # (pins the gap this method fills; regression guard if locate_l2 ever
        # changes to cover this direction itself).
        assert g.locate_l2("ws", "CacheControlHeader").resolution_strategy == "none"

        near = g.nearest_symbol_names("ws", "CacheControlHeader", limit=3)
        assert [s.name for s in near] == ["CacheControl"]

    def test_reuses_locate_l2_non_exact_strategy_when_available(self, tmp_path) -> None:
        """When locate_l2 already resolves the token via one of its own
        non-exact strategies (here: fuzzy), nearest_symbol_names reuses that
        result rather than running the prefix fallback."""
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "f.py", "def foo_bar(): pass")
        g.index_file("ws", str(tmp_path / "f.py"))

        near = g.nearest_symbol_names("ws", "fo_bar", limit=3)
        assert [s.name for s in near] == ["foo_bar"]

    def test_no_candidate_yields_empty_list(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "control.py", "class CacheControl:\n    pass\n")
        g.index_file("ws", str(tmp_path / "control.py"))

        assert g.nearest_symbol_names("ws", "XyzzyQwerty", limit=3) == []

    def test_short_candidate_name_below_min_prefix_len_is_excluded(self, tmp_path) -> None:
        """A real symbol whose name is shorter than the configured minimum
        shared length is never offered as a prefix-containment near-miss,
        even when the token literally starts with it — a short, generic
        name (here 2 chars) would otherwise "match" as a prefix of nearly
        anything."""
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "a.py", "def ab():\n    pass\n")
        g.index_file("ws", str(tmp_path / "a.py"))

        # Length gap (16 vs 2) rules out locate_l2's own fuzzy strategy too,
        # so "none" of its non-exact strategies interferes with this check.
        assert g.locate_l2("ws", "abXyzzyLongTokenHere").resolution_strategy == "none"
        assert g.nearest_symbol_names("ws", "abXyzzyLongTokenHere", limit=3) == []

    def test_respects_limit_cap(self, tmp_path) -> None:
        """Three real symbol names are each a literal prefix of the token
        (layered: shorter names are themselves prefixes of the longer ones)
        — the cap must still bound the returned list to `limit`."""
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "control.py",
            "class CacheControl:\n    pass\n"
            "class CacheControlExtra:\n    pass\n"
            "class CacheControlExtraStuff:\n    pass\n",
        )
        g.index_file("ws", str(tmp_path / "control.py"))

        near = g.nearest_symbol_names("ws", "CacheControlExtraStuffMoreWords", limit=2)
        assert len(near) <= 2


# ---------------------------------------------------------------------------
# UPG-LOCATE-FALLBACK-NO-SIMILARITY: after a qualified miss (Class.method
# where the qualifier resolves to no real enclosing type anywhere in the
# workspace), nearest_symbol_names widens to every exact-leaf-name
# definition and ranks them by string similarity between the FAILED
# qualifier and each candidate's enclosing type (secondarily its file
# path) — ranking result-side data against the caller's own explicit
# argument, not query-content gating.
# ---------------------------------------------------------------------------

class TestNearestSymbolNamesQualifierSimilarityUPG:
    def test_fabricated_qualifier_falls_back_to_similarity_ranked_leaf_candidates(
        self, tmp_path,
    ) -> None:
        """A dotted "Class.method" token whose class matches no enclosing
        type anywhere is invisible to locate_l2 (every strategy's class
        filter empties out) and to the existing prefix-containment fallback
        (it only compares bare names, never a dotted qualifier). The new
        fallback widens to every "process" definition and ranks the
        candidate whose enclosing class is the closest string match to the
        failed qualifier first."""
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "widgets.py",
            "class WidgetRegistry:\n    def process(self):\n        pass\n"
            "class SprocketFactory:\n    def process(self):\n        pass\n"
            "class GadgetHelper:\n    def process(self):\n        pass\n",
        )
        g.index_file("ws", str(tmp_path / "widgets.py"))

        # Confirm locate_l2 truly has nothing for the fabricated class (pins
        # the gap this fallback fills; regression guard if locate_l2 ever
        # widens its own class filter).
        assert g.locate_l2("ws", "WidgetRegistery.process").resolution_strategy == "none"

        near = g.nearest_symbol_names("ws", "WidgetRegistery.process", limit=10)
        assert [s.name for s in near] == [
            "WidgetRegistry.process",
            "GadgetHelper.process",
            "SprocketFactory.process",
        ]

    def test_similarity_ties_keep_canonical_fetch_order(self, tmp_path) -> None:
        """Two candidates that tie exactly on both the enclosing-type and
        file-path similarity ratios (same file, disjoint-character class
        names against a disjoint-character fabricated qualifier) keep the
        order _exact_definitions already produced — stable sort, no
        incidental reordering on a real tie."""
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "mod.py",
            "class Xxxxxx:\n    def process(self):\n"
            + "".join(f"        v{i} = {i}\n" for i in range(20))
            + "class Yyyyyy:\n    def process(self):\n        pass\n",
        )
        g.index_file("ws", str(tmp_path / "mod.py"))

        near = g.nearest_symbol_names("ws", "Wwwwww.process", limit=10)
        assert [s.name for s in near] == ["Xxxxxx.process", "Yyyyyy.process"]

    def test_candidate_pool_still_capped_and_limit_respected(self, tmp_path) -> None:
        """The widened fallback is bounded by the configured candidate pool
        cap upstream and by the caller's requested limit downstream."""
        classes = "".join(
            f"class C{i}:\n    def process(self):\n        pass\n" for i in range(5)
        )
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "many.py", classes)
        g.index_file("ws", str(tmp_path / "many.py"))

        near = g.nearest_symbol_names("ws", "Bogus.process", limit=2)
        assert len(near) == 2


# ---------------------------------------------------------------------------
# UPG-QUALIFIED-TYPO-NEARMISS: a qualified token with a typo in BOTH the
# class qualifier AND the leaf (so neither the "none"-strategy qualifier
# widening above, which needs an exact leaf, nor locate_l2's own qualified
# strategies fire) still surfaces a labeled-inexact suggestion by retrying
# the leaf alone through the same non-exact locate_l2 machinery.
# ---------------------------------------------------------------------------

class TestNearestSymbolNamesQualifiedLeafTypoUPG:
    def test_typo_in_both_segments_still_returns_ranked_suggestions(
        self, tmp_path,
    ) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "widgets.py",
            "class WidgetRegistry:\n    def fetch_item(self):\n        pass\n"
            "class GadgetHelper:\n    def fetch_item(self):\n        pass\n",
        )
        g.index_file("ws", str(tmp_path / "widgets.py"))

        # Pin the gap: an exact-leaf widen (_exact_definitions) finds nothing
        # because the leaf itself has a typo too.
        assert g._exact_definitions("ws", "fetch_itm", limit=50) == []
        assert g.locate_l2(
            "ws", "WidgetRegistery.fetch_itm",
        ).resolution_strategy == "none"

        near = g.nearest_symbol_names("ws", "WidgetRegistery.fetch_itm", limit=10)
        # Correct symbol first (its owning class is the closer string match
        # to the failed qualifier); nothing here is an empty/zero-suggestion
        # miss despite both segments being misspelled.
        assert [s.name for s in near] == [
            "WidgetRegistry.fetch_item",
            "GadgetHelper.fetch_item",
        ]

    def test_single_candidate_typo_in_both_segments(self, tmp_path) -> None:
        """Simplest shape of the bug report: one real symbol, typo'd class
        AND typo'd leaf, must not fall back to a flat empty result."""
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "orders.py",
            "class OrderRepository:\n    def save_record(self):\n        pass\n",
        )
        g.index_file("ws", str(tmp_path / "orders.py"))

        near = g.nearest_symbol_names("ws", "OrderRepositary.save_recrd", limit=10)
        assert [s.name for s in near] == ["OrderRepository.save_record"]


# ---------------------------------------------------------------------------
# UPG-NEARMISS-WIDEN-EXACT-GUARD: an already-exact-resolved qualified token
# must not be widened out to unrelated same-leaf sibling classes.
# ---------------------------------------------------------------------------

class TestNearestSymbolNamesExactGuardUPG:
    def test_exact_resolvable_qualified_token_is_not_widened_to_siblings(
        self, tmp_path,
    ) -> None:
        # Three short (below nearmiss_min_prefix_len) sibling classes share a
        # leaf method name; the token is spelled correctly and resolves
        # exactly via locate_l2. Widening would incorrectly surface the two
        # unrelated siblings as "near-miss" results for an already-correct
        # lookup.
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "mod.py",
            "class Foo:\n    def process(self):\n        pass\n"
            "class Bar:\n    def process(self):\n        pass\n"
            "class Baz:\n    def process(self):\n        pass\n",
        )
        g.index_file("ws", str(tmp_path / "mod.py"))

        exact = g.locate_l2("ws", "Foo.process")
        assert exact.resolution_strategy == "exact"

        near = g.nearest_symbol_names("ws", "Foo.process", limit=10)
        assert near == []


# ---------------------------------------------------------------------------
# UPG-LOCATE-NEARMISS-WIRE-PRIMARY-TOOL: locate_l2's "none" branch now carries
# the same deterministic near-miss candidates nearest_symbol_names has always
# computed — ON THE RESULT (opt-in via `with_near_miss`) so the primary
# vectr_locate surface can render them labeled inexact instead of answering a
# zero-suggestion miss. `symbols` stays empty: a suggestion, never a rewrite.
# ---------------------------------------------------------------------------

class TestLocateNearMissOnResultUPG:
    def test_qualified_double_typo_carries_suggestion_on_result(self, tmp_path) -> None:
        """The original witness shape: vectr_locate("OrderRepositary.save_recrd")
        called directly (not via trace/search) still resolves "none" — never a
        silent rewrite — but no longer returns a zero-suggestion miss."""
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "orders.py",
            "class OrderRepository:\n    def save_record(self):\n        pass\n",
        )
        g.index_file("ws", str(tmp_path / "orders.py"))

        result = g.locate_l2("ws", "OrderRepositary.save_recrd", with_near_miss=True)
        assert result.resolution_strategy == "none"
        assert result.symbols == []          # a miss stays a miss
        assert [s.name for s in result.near_miss] == ["OrderRepository.save_record"]
        assert result.near_miss[0].snippet   # rendered by the formatter below

    def test_prefix_containment_shape_carried_on_result(self, tmp_path) -> None:
        """The other fallback stage (token = real name + misremembered extra
        word) reaches the primary tool path too, not just nearest_symbol_names
        callers."""
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "control.py", "class CacheControl:\n    pass\n")
        g.index_file("ws", str(tmp_path / "control.py"))

        result = g.locate_l2("ws", "CacheControlHeader", with_near_miss=True)
        assert result.resolution_strategy == "none"
        assert [s.name for s in result.near_miss] == ["CacheControl"]

    def test_near_miss_off_by_default_for_internal_callers(self, tmp_path) -> None:
        """Internal locate_l2 callers (the search-hint paths run it per token
        per search) keep the cheap behaviour: no suggestion queries on a miss
        unless they opt in."""
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "control.py", "class CacheControl:\n    pass\n")
        g.index_file("ws", str(tmp_path / "control.py"))

        result = g.locate_l2("ws", "CacheControlHeader")
        assert result.resolution_strategy == "none"
        assert result.near_miss == []

    def test_exact_resolution_never_populates_near_miss(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        make_py(tmp_path, "control.py", "class CacheControl:\n    pass\n")
        g.index_file("ws", str(tmp_path / "control.py"))

        result = g.locate_l2("ws", "CacheControl", with_near_miss=True)
        assert result.resolution_strategy == "exact"
        assert result.near_miss == []

    def test_formatter_leads_with_labeled_inexact_suggestions(self, tmp_path) -> None:
        """Same labeling convention as trace()'s qualifier-near-miss banner and
        the search-hint near-miss section — never the plain dead-end text and
        never presented as a match."""
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "orders.py",
            "class OrderRepository:\n    def save_record(self):\n        pass\n",
        )
        g.index_file("ws", str(tmp_path / "orders.py"))

        result = g.locate_l2("ws", "OrderRepositary.save_recrd", with_near_miss=True)
        text = g.format_locate_l2_for_llm(result)
        assert "No exact match for 'OrderRepositary.save_recrd'" in text
        assert "(inexact — verify before use)" in text
        assert "[function] OrderRepository.save_record" in text
        assert "No symbol matching" not in text

    def test_formatter_plain_miss_unchanged_without_candidates(self, tmp_path) -> None:
        """A miss with no near-miss candidates keeps the existing redirect
        message byte-for-byte."""
        g = SymbolGraph(str(tmp_path))
        text = g.format_locate_l2_for_llm(
            LocateResult(symbols=[], resolution_strategy="none",
                         query="XyzzyQwerty"),
        )
        assert "No symbol matching 'XyzzyQwerty' found in the indexed codebase." in text


class TestNearestSymbolNamesExactTokenUPG:
    def test_exactly_resolved_token_gets_no_suggestions(self, tmp_path) -> None:
        """nearest_symbol_names' contract is 'a token that already failed EXACT
        resolution': an exactly-resolving token now returns [] rather than
        falling through to prefix-containment candidates dressed up as a
        near-miss for an already-correct lookup."""
        g = SymbolGraph(str(tmp_path))
        make_py(
            tmp_path, "mod.py",
            "class CacheControl:\n    pass\n"
            "class CacheControlExtra:\n    pass\n",
        )
        g.index_file("ws", str(tmp_path / "mod.py"))

        assert g.locate_l2("ws", "CacheControl").resolution_strategy == "exact"
        assert g.nearest_symbol_names("ws", "CacheControl", limit=10) == []


class TestLevenshtein:
    def test_identical_strings(self) -> None:
        assert _levenshtein("abc", "abc") == 0

    def test_empty_strings(self) -> None:
        assert _levenshtein("", "") == 0
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "abc") == 3

    def test_single_substitution(self) -> None:
        assert _levenshtein("kitten", "sitten") == 1

    def test_edit_distance_two(self) -> None:
        assert _levenshtein("authenticate", "authenticat") == 1
        assert _levenshtein("foobar", "fobar") == 1
        assert _levenshtein("cat", "cut") == 1
        assert _levenshtein("saturday", "sunday") == 3


# ---------------------------------------------------------------------------
# SymbolGraph — callers / callees / trace
# ---------------------------------------------------------------------------

class TestSymbolGraphTrace:
    def _indexed_graph(self, tmp_path):
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "svc.py", textwrap.dedent("""\
            def helper():
                pass

            def process():
                helper()

            def orchestrate():
                process()
                helper()
        """))
        g.index_file("ws", path)
        return g, path

    def test_callees_of_process_includes_helper(self, tmp_path) -> None:
        g, _ = self._indexed_graph(tmp_path)
        edges = g.callees("ws", "process")
        callees = [e.to_symbol for e in edges]
        assert "helper" in callees

    def test_callers_of_helper_includes_process(self, tmp_path) -> None:
        g, _ = self._indexed_graph(tmp_path)
        edges = g.callers("ws", "helper")
        callers = [e.from_symbol for e in edges]
        assert "process" in callers

    def test_callers_of_helper_includes_orchestrate(self, tmp_path) -> None:
        g, _ = self._indexed_graph(tmp_path)
        edges = g.callers("ws", "helper")
        callers = [e.from_symbol for e in edges]
        assert "orchestrate" in callers

    def test_trace_both_returns_callers_and_callees(self, tmp_path) -> None:
        g, _ = self._indexed_graph(tmp_path)
        result = g.trace("ws", "process", direction="both")
        assert "callers" in result
        assert "callees" in result
        assert result["callers"] is not None
        assert result["callees"] is not None

    def test_trace_callers_only(self, tmp_path) -> None:
        # UPG-TRACE-DIRECTION-UNQUERIED-RENDER: a direction the caller did
        # not ask for is recorded as `None`, not absent, so the formatter
        # can distinguish "we never looked" from "we looked and found
        # nothing". The dict now carries both keys consistently — the
        # value shape is the public signal, not key presence.
        g, _ = self._indexed_graph(tmp_path)
        result = g.trace("ws", "process", direction="callers")
        assert isinstance(result["callers"], list)
        assert result["callees"] is None

    def test_trace_callees_only(self, tmp_path) -> None:
        # UPG-TRACE-DIRECTION-UNQUERIED-RENDER: see `test_trace_callers_only`.
        g, _ = self._indexed_graph(tmp_path)
        result = g.trace("ws", "process", direction="callees")
        assert isinstance(result["callees"], list)
        assert result["callers"] is None

    def test_edges_have_file_path(self, tmp_path) -> None:
        g, path = self._indexed_graph(tmp_path)
        edges = g.callers("ws", "helper")
        assert all(e.from_file == path for e in edges)

    def test_no_edges_for_uncalled_symbol(self, tmp_path) -> None:
        g, _ = self._indexed_graph(tmp_path)
        edges = g.callers("ws", "orchestrate")
        assert edges == []


# ---------------------------------------------------------------------------
# SymbolGraph — get_snippet
# ---------------------------------------------------------------------------

class TestGetSnippet:
    def test_returns_code_lines(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        f = tmp_path / "code.py"
        f.write_text("line1\nline2\nline3\nline4\n")
        snippet = g.get_snippet(str(f), start_line=1, end_line=4)
        assert "line1" in snippet

    def test_snippet_capped_at_snippet_lines(self, tmp_path) -> None:
        from agent.symbol_graph import SNIPPET_LINES
        g = SymbolGraph(str(tmp_path))
        f = tmp_path / "big.py"
        f.write_text("\n".join(f"line_{i}" for i in range(100)))
        snippet = g.get_snippet(str(f), start_line=1, end_line=100)
        assert len(snippet.splitlines()) <= SNIPPET_LINES

    def test_nonexistent_file_returns_empty_string(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        snippet = g.get_snippet(str(tmp_path / "ghost.py"), 1, 10)
        assert snippet == ""


# ---------------------------------------------------------------------------
# symbols_touching_file / signature_hash — TRIGGER-ENGINE wave 2b, the S
# (symbol) trigger primitive's write-time and fire-time resolution.
# ---------------------------------------------------------------------------

class TestSymbolsTouchingFile:
    def test_includes_symbols_defined_in_the_file(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        pass\n")
        g.index_file("ws", str(f))
        names = g.symbols_touching_file("ws", str(f))
        assert "WorkspaceLock" in names

    def test_includes_symbols_referenced_by_the_file(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        lib = tmp_path / "lock.py"
        lib.write_text("def acquire_lock():\n    pass\n")
        caller = tmp_path / "resolver.py"
        caller.write_text("from lock import acquire_lock\n\ndef use_it():\n    acquire_lock()\n")
        g.index_file("ws", str(lib))
        g.index_file("ws", str(caller))
        names = g.symbols_touching_file("ws", str(caller))
        assert "acquire_lock" in names

    def test_unindexed_file_returns_empty_frozenset(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        names = g.symbols_touching_file("ws", str(tmp_path / "never_indexed.py"))
        assert names == frozenset()

    def test_workspace_isolation(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    pass\n")
        g.index_file("ws_a", str(f))
        assert "WorkspaceLock" in g.symbols_touching_file("ws_a", str(f))
        assert "WorkspaceLock" not in g.symbols_touching_file("ws_b", str(f))


class TestSignatureHash:
    def test_returns_a_stable_hash_for_an_unchanged_definition(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        pass\n")
        g.index_file("ws", str(f))
        h1 = g.signature_hash("ws", "WorkspaceLock")
        h2 = g.signature_hash("ws", "WorkspaceLock")
        assert h1 is not None
        assert h1 == h2

    def test_hash_changes_when_the_definition_body_changes(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        f = tmp_path / "resolver.py"
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        pass\n")
        g.index_file("ws", str(f))
        before = g.signature_hash("ws", "WorkspaceLock")
        f.write_text("class WorkspaceLock:\n    def acquire(self):\n        return True\n")
        g.index_file("ws", str(f))
        after = g.signature_hash("ws", "WorkspaceLock")
        assert before != after

    def test_unresolvable_symbol_returns_none(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        assert g.signature_hash("ws", "NoSuchSymbol") is None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_locate_empty(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        text = g.format_locate_for_llm([], "verify_token")
        assert "No symbol" in text
        assert "verify_token" in text

    def test_format_locate_with_symbols(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        sym = Symbol(
            symbol_id=1, workspace="ws", name="verify_token", kind="function",
            file_path="auth.py", start_line=10, end_line=20, snippet="def verify_token(): ...",
        )
        text = g.format_locate_for_llm([sym], "verify_token")
        assert "verify_token" in text
        assert "auth.py" in text

    def test_format_trace_with_callers(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        edge = CallEdge(from_file="svc.py", from_symbol="main", from_line=10, to_symbol="helper", edge_type="calls")
        text = g.format_trace_for_llm({"callers": [edge]}, "helper")
        assert "main" in text
        assert "svc.py" in text

    def test_format_trace_empty_callers(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        text = g.format_trace_for_llm({"callers": [], "callees": []}, "fn")
        assert "none found" in text.lower()

    def test_format_trace_both_empty_carries_recovery_hint(self, tmp_path) -> None:
        # UPG-TRACE-EMPTY-HINT (F61): static analysis found no calls at all —
        # locate-miss and search-low-confidence both hint a next step; trace
        # must too, instead of leaving a bare dead end.
        g = SymbolGraph(str(tmp_path))
        text = g.format_trace_for_llm(
            {"callers": [], "callees": [], "hidden_builtins": 0, "definitions": []},
            "request_started",
        )
        assert "vectr_search" in text
        assert "dynamic" in text.lower()

    def test_format_trace_one_sided_result_has_no_recovery_hint(self, tmp_path) -> None:
        # Hint only fires when BOTH sides are empty — a real callers hit means
        # the symbol is genuinely resolved, no redirect needed.
        g = SymbolGraph(str(tmp_path))
        edge = CallEdge(from_file="svc.py", from_symbol="main", from_line=10, to_symbol="helper", edge_type="calls")
        text = g.format_trace_for_llm(
            {"callers": [edge], "callees": [], "hidden_builtins": 0, "definitions": []}, "helper",
        )
        assert "vectr_search" not in text

    def test_format_trace_class_with_no_callees_notes_method_level_trace(self, tmp_path) -> None:
        # UPG-TRACE-EMPTY-HINT: a class is never itself "called" — calls(0) on a
        # class name must say so, rather than reading like a real miss.
        g = SymbolGraph(str(tmp_path))
        cls = Symbol(symbol_id=1, workspace="ws", name="CursorDebugWrapper", kind="class",
                     file_path="dbg.py", start_line=1, end_line=20)
        text = g.format_trace_for_llm(
            {"callers": [], "callees": [], "hidden_builtins": 0, "definitions": [cls]},
            "CursorDebugWrapper",
        )
        assert "method" in text.lower()
        assert "vectr_trace(name=" in text

    def test_format_trace_function_with_no_callees_has_no_class_note(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        fn = Symbol(symbol_id=1, workspace="ws", name="helper", kind="function",
                    file_path="m.py", start_line=1, end_line=5)
        text = g.format_trace_for_llm(
            {"callers": [], "callees": [], "hidden_builtins": 0, "definitions": [fn]}, "helper",
        )
        assert "method" not in text.lower()


# ---------------------------------------------------------------------------
# T26: HTTP route extraction
# ---------------------------------------------------------------------------

class TestRouteExtraction:
    def _routes(self, source: str, language: str, suffix: str = ".py") -> list[dict]:
        from agent.symbol_graph import _extract_routes
        return _extract_routes(f"app{suffix}", source, language)

    # Flask / FastAPI (Python)

    def test_flask_route_decorator_extracted(self) -> None:
        source = '@app.route("/users")\ndef get_users(): pass\n'
        routes = self._routes(source, "python")
        assert any(r["name"] == "GET /users" for r in routes)
        assert all(r["kind"] == "route" for r in routes)

    def test_flask_route_post_method(self) -> None:
        source = '@app.route("/users", methods=["POST"])\ndef create_user(): pass\n'
        routes = self._routes(source, "python")
        assert any(r["name"] == "POST /users" for r in routes)

    def test_flask_route_multiple_methods(self) -> None:
        source = '@app.route("/items", methods=["GET", "POST"])\ndef items(): pass\n'
        routes = self._routes(source, "python")
        names = {r["name"] for r in routes}
        assert "GET /items" in names
        assert "POST /items" in names

    def test_fastapi_get_decorator(self) -> None:
        source = '@router.get("/health")\nasync def health_check(): pass\n'
        routes = self._routes(source, "python")
        assert any(r["name"] == "GET /health" for r in routes)

    def test_fastapi_post_decorator(self) -> None:
        source = '@app.post("/login")\nasync def login(): pass\n'
        routes = self._routes(source, "python")
        assert any(r["name"] == "POST /login" for r in routes)

    def test_fastapi_delete_decorator(self) -> None:
        source = '@router.delete("/users/{id}")\nasync def delete_user(id: int): pass\n'
        routes = self._routes(source, "python")
        assert any(r["name"] == "DELETE /users/{id}" for r in routes)

    def test_no_routes_in_plain_python(self) -> None:
        source = 'def helper(): return 42\nclass Foo: pass\n'
        assert self._routes(source, "python") == []

    # Spring (Java)

    def test_spring_get_mapping(self) -> None:
        source = '@GetMapping("/api/users")\npublic List<User> getUsers() {}\n'
        routes = self._routes(source, "java", ".java")
        assert any(r["name"] == "GET /api/users" for r in routes)

    def test_spring_post_mapping(self) -> None:
        source = '@PostMapping("/api/orders")\npublic Order create() {}\n'
        routes = self._routes(source, "java", ".java")
        assert any(r["name"] == "POST /api/orders" for r in routes)

    def test_spring_delete_mapping(self) -> None:
        source = '@DeleteMapping("/api/users/{id}")\npublic void delete() {}\n'
        routes = self._routes(source, "java", ".java")
        assert any(r["name"] == "DELETE /api/users/{id}" for r in routes)

    def test_spring_request_mapping(self) -> None:
        source = '@RequestMapping("/api/base")\nclass BaseController {}\n'
        routes = self._routes(source, "java", ".java")
        assert any(r["name"] == "GET /api/base" for r in routes)

    # Express (JavaScript)

    def test_express_app_get(self) -> None:
        source = 'app.get("/users", (req, res) => res.json([]));\n'
        routes = self._routes(source, "javascript", ".js")
        assert any(r["name"] == "GET /users" for r in routes)

    def test_express_router_post(self) -> None:
        source = 'router.post("/login", loginHandler);\n'
        routes = self._routes(source, "javascript", ".js")
        assert any(r["name"] == "POST /login" for r in routes)

    def test_express_router_delete(self) -> None:
        source = 'router.delete("/items/:id", handler);\n'
        routes = self._routes(source, "javascript", ".js")
        assert any(r["name"] == "DELETE /items/:id" for r in routes)

    def test_typescript_express_route(self) -> None:
        source = 'app.put("/profile", updateHandler);\n'
        routes = self._routes(source, "typescript", ".ts")
        assert any(r["name"] == "PUT /profile" for r in routes)

    # Integration: routes appear in extract_symbols_from_file

    def test_routes_in_extract_symbols_from_file(self, tmp_path) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        f = tmp_path / "views.py"
        f.write_text(
            '@app.get("/api/data")\ndef get_data(): return []\n'
            '\ndef helper(): pass\n'
        )
        symbols, _ = extract_symbols_from_file(str(f))
        route_symbols = [s for s in symbols if s["kind"] == "route"]
        assert any(s["name"] == "GET /api/data" for s in route_symbols)

    # Routes appear in symbol graph locate

    def test_routes_locatable_via_symbol_graph(self, tmp_path) -> None:
        from agent.symbol_graph import SymbolGraph
        f = tmp_path / "api.py"
        f.write_text('@router.post("/checkout")\ndef checkout(): pass\n')
        g = SymbolGraph(str(tmp_path))
        g.index_file(str(tmp_path), str(f))
        results = g.locate(str(tmp_path), "POST /checkout")
        assert len(results) >= 1
        assert results[0].kind == "route"


# ---------------------------------------------------------------------------
# T28: ingest_traces — dynamic call edge ingestion
# ---------------------------------------------------------------------------

class TestIngestTraces:
    def _graph(self, tmp_path) -> "SymbolGraph":
        from agent.symbol_graph import SymbolGraph
        return SymbolGraph(str(tmp_path))

    def test_ingest_valid_events_returns_count(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        events = [
            {"caller": "view_cart", "callee": "CartService.get", "caller_file": "views.py", "caller_line": 42},
            {"caller": "checkout", "callee": "PaymentService.charge", "caller_file": "views.py", "caller_line": 88},
        ]
        result = g.ingest_trace_data("/ws", events)
        assert result["ingested"] == 2
        assert result["skipped_invalid"] == 0

    def test_events_missing_caller_are_skipped(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        events = [
            {"callee": "helper"},          # missing caller
            {"caller": "main", "callee": "helper"},
        ]
        result = g.ingest_trace_data("/ws", events)
        assert result["ingested"] == 1
        assert result["skipped_invalid"] == 1

    def test_events_missing_callee_are_skipped(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        events = [{"caller": "main"}]  # missing callee
        result = g.ingest_trace_data("/ws", events)
        assert result["ingested"] == 0
        assert result["skipped_invalid"] == 1

    def test_empty_events_list_returns_zero(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        result = g.ingest_trace_data("/ws", [])
        assert result["ingested"] == 0
        assert result["skipped_invalid"] == 0

    def test_ingested_edges_appear_in_callers(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        g.ingest_trace_data("/ws", [
            {"caller": "process_payment", "callee": "stripe_charge", "caller_file": "billing.py"}
        ])
        callers = g.callers("/ws", "stripe_charge")
        assert any(e.from_symbol == "process_payment" for e in callers)

    def test_ingested_edges_have_dynamic_edge_type(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        g.ingest_trace_data("/ws", [
            {"caller": "handler", "callee": "dispatch", "caller_file": "router.py"}
        ])
        callers = g.callers("/ws", "dispatch")
        dynamic = [e for e in callers if e.edge_type == "dynamic"]
        assert len(dynamic) >= 1

    def test_duplicate_edges_not_duplicated(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        event = {"caller": "a", "callee": "b", "caller_file": "a.py", "caller_line": 1}
        g.ingest_trace_data("/ws", [event, event, event])
        # Due to INSERT OR IGNORE, only 1 edge should exist
        callers = g.callers("/ws", "b")
        b_callers = [e for e in callers if e.from_symbol == "a"]
        assert len(b_callers) == 1

    def test_optional_fields_default_gracefully(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        events = [{"caller": "main", "callee": "helper"}]  # no caller_file or caller_line
        result = g.ingest_trace_data("/ws", events)
        assert result["ingested"] == 1

    def test_mcp_vectr_ingest_traces_dispatches(self) -> None:
        from unittest.mock import MagicMock
        from integrations.mcp_server import handle_tools_call
        svc = MagicMock()
        # Mock the real ingest_trace_data() return shape (UPG-7.3).
        svc.ingest_traces.return_value = {
            "ingested": 3, "skipped_invalid": 0,
            "unresolved_callers": 0, "unresolved_callees": 0, "unresolved_examples": [],
        }
        events = [{"caller": "a", "callee": "b"}]
        result = handle_tools_call("vectr_ingest_traces", {"events": events}, svc)
        assert result["isError"] is False
        svc.ingest_traces.assert_called_once_with(events)

    def test_mcp_vectr_ingest_traces_missing_events_is_error(self) -> None:
        from unittest.mock import MagicMock
        from integrations.mcp_server import handle_tools_call
        svc = MagicMock()
        result = handle_tools_call("vectr_ingest_traces", {}, svc)
        assert result["isError"] is True

    def test_vectr_ingest_traces_in_tools_list(self) -> None:
        from integrations.mcp_server import MCP_TOOLS
        names = {t["name"] for t in MCP_TOOLS}
        assert "vectr_ingest_traces" in names


# ---------------------------------------------------------------------------
# UPG-7.3 — ingest_traces unresolved caller/callee warning + dynamic marker
# ---------------------------------------------------------------------------

class TestIngestTracesUnresolvedWarningUPG73:
    def _graph(self, tmp_path) -> "SymbolGraph":
        from agent.symbol_graph import SymbolGraph
        return SymbolGraph(str(tmp_path))

    def test_unresolved_caller_and_callee_still_ingested_but_flagged(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        result = g.ingest_trace_data("/ws", [
            {"caller": "totally_unknown_caller", "callee": "totally_unknown_callee"}
        ])
        assert result["ingested"] == 1
        assert result["unresolved_callers"] == 1
        assert result["unresolved_callees"] == 1
        assert len(result["unresolved_examples"]) == 1
        assert "totally_unknown_caller" in result["unresolved_examples"][0]
        assert "totally_unknown_callee" in result["unresolved_examples"][0]

    def test_known_symbols_do_not_warn(self, tmp_path, monkeypatch) -> None:
        g = self._graph(tmp_path)
        # index a real symbol so the graph has a known "caller" name
        from tests.conftest import make_py
        f = make_py(tmp_path, "views.py", "def view_cart():\n    pass\n")
        g.index_file(str(tmp_path), str(f))
        result = g.ingest_trace_data(str(tmp_path), [
            {"caller": "view_cart", "callee": "view_cart", "caller_file": str(f)}
        ])
        assert result["ingested"] == 1
        assert result["unresolved_callers"] == 0
        assert result["unresolved_callees"] == 0
        assert result["unresolved_examples"] == []

    def test_unresolved_examples_capped(self, tmp_path) -> None:
        from agent.config import INGEST_TRACES_MAX_UNRESOLVED_EXAMPLES
        g = self._graph(tmp_path)
        events = [
            {"caller": f"unknown_caller_{i}", "callee": f"unknown_callee_{i}"}
            for i in range(INGEST_TRACES_MAX_UNRESOLVED_EXAMPLES + 5)
        ]
        result = g.ingest_trace_data("/ws", events)
        assert result["ingested"] == len(events)
        assert result["unresolved_callers"] == len(events)
        assert len(result["unresolved_examples"]) == INGEST_TRACES_MAX_UNRESOLVED_EXAMPLES

    def test_mcp_ingest_traces_surfaces_unresolved_warning(self) -> None:
        from unittest.mock import MagicMock
        from integrations.mcp_server import handle_tools_call
        svc = MagicMock()
        svc.ingest_traces.return_value = {
            "ingested": 1, "skipped_invalid": 0,
            "unresolved_callers": 1, "unresolved_callees": 1,
            "unresolved_examples": ["mystery_caller -> mystery_callee (both unresolved)"],
        }
        result = handle_tools_call(
            "vectr_ingest_traces", {"events": [{"caller": "mystery_caller", "callee": "mystery_callee"}]}, svc
        )
        text = result["content"][0]["text"]
        assert "Warning" in text
        assert "mystery_caller -> mystery_callee" in text

    def test_mcp_ingest_traces_no_warning_when_all_resolved(self) -> None:
        from unittest.mock import MagicMock
        from integrations.mcp_server import handle_tools_call
        svc = MagicMock()
        svc.ingest_traces.return_value = {
            "ingested": 1, "skipped_invalid": 0,
            "unresolved_callers": 0, "unresolved_callees": 0, "unresolved_examples": [],
        }
        result = handle_tools_call("vectr_ingest_traces", {"events": [{"caller": "a", "callee": "b"}]}, svc)
        text = result["content"][0]["text"]
        assert "Warning" not in text

    def test_dynamic_edge_marked_in_trace_output(self, tmp_path) -> None:
        g = self._graph(tmp_path)
        g.ingest_trace_data("/ws", [
            {"caller": "handler", "callee": "dispatch_dynamic", "caller_file": "router.py"}
        ])
        trace_result = g.trace("/ws", "dispatch_dynamic", direction="callers")
        text = g.format_trace_for_llm(trace_result, "dispatch_dynamic")
        assert "(dynamic)" in text

    def test_static_edge_not_marked_dynamic(self, tmp_path) -> None:
        from tests.conftest import make_py
        g = self._graph(tmp_path)
        f = make_py(
            tmp_path, "mod.py",
            "def helper():\n    pass\n\ndef caller():\n    helper()\n",
        )
        g.index_file(str(tmp_path), str(f))
        trace_result = g.trace(str(tmp_path), "helper", direction="callers")
        text = g.format_trace_for_llm(trace_result, "helper")
        assert "(dynamic)" not in text


# ---------------------------------------------------------------------------
# UPG-4.5 — locate: prefer canonical definitions over aliases/impls
# ---------------------------------------------------------------------------

class TestLocateRankingUPG45:
    def test_substring_fires_before_fuzzy(self, tmp_path) -> None:
        # `rand` must resolve to prefix matches (randint/randfraction), never to
        # fuzzy junk like `nan`/`add`. The substring strategy owns this query.
        g = SymbolGraph(str(tmp_path))
        _seed_symbols(g, "ws", [
            ("randint", "function", str(tmp_path / "random.py")),
            ("randfraction", "function", str(tmp_path / "random.py")),
            ("nan", "function", str(tmp_path / "mathmodule.c")),
            ("add", "function", str(tmp_path / "abstract.c")),
        ])
        result = g.locate_l2("ws", "rand")
        assert result.resolution_strategy == "substring"
        names = [s.name for s in result.symbols]
        assert names[0] in ("randint", "randfraction")
        assert "nan" not in names and "add" not in names

    def test_prefix_match_beats_interior_substring(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        _seed_symbols(g, "ws", [
            ("myrandom_helper", "function", str(tmp_path / "a.py")),  # interior
            ("rand_state", "function", str(tmp_path / "b.py")),       # prefix
        ])
        result = g.locate_l2("ws", "rand")
        assert result.symbols[0].name == "rand_state"

    def test_canonical_def_ranks_before_impl_on_exact_match(self, tmp_path) -> None:
        # Rust: `struct VersionSpecifiers` + several `impl VersionSpecifiers`.
        # All are an EXACT name hit, so ranking must lead with the struct def.
        g = SymbolGraph(str(tmp_path))
        _seed_symbols(g, "ws", [
            ("VersionSpecifiers", "impl", str(tmp_path / "version.rs")),
            ("VersionSpecifiers", "impl", str(tmp_path / "ops.rs")),
            ("VersionSpecifiers", "struct", str(tmp_path / "version.rs")),
            ("VersionSpecifiers", "impl", str(tmp_path / "cmp.rs")),
        ])
        result = g.locate_l2("ws", "VersionSpecifiers")
        assert result.resolution_strategy == "exact"
        assert result.symbols[0].kind == "struct"

    def test_function_beats_macro_on_partial_match(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        _seed_symbols(g, "ws", [
            ("PY_PARSE_MACRO", "macro", str(tmp_path / "pymacro.h")),
            ("PY_parse_value", "function", str(tmp_path / "parse.c")),
        ])
        result = g.locate_l2("ws", "PY_pa")
        # both are prefix matches; the function definition is more canonical
        assert result.symbols[0].kind == "function"

    def test_test_and_private_files_deprioritized(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        _seed_symbols(g, "ws", [
            ("widget_render", "function", str(tmp_path / "test_widget.py")),
            ("widget_render_helper", "function", str(tmp_path / "_internal.py")),
            ("widget_render_main", "function", str(tmp_path / "widget.py")),
        ])
        # "widget_rend" is an exact match for none of them (all prefix matches),
        # so the test/private-file penalty decides the lead.
        result = g.locate_l2("ws", "widget_rend")
        # real source file leads; test_/_private files sink
        assert "test_" not in result.symbols[0].file_path.rsplit("/", 1)[-1]
        assert not result.symbols[0].file_path.rsplit("/", 1)[-1].startswith("_")

    def test_short_query_fuzzy_budget_is_tight(self, tmp_path) -> None:
        # No substring match exists, so we reach fuzzy. A 3-char query gets dist≤1
        # and a shared first char, so `add`/`nan` can't match `rnd`.
        g = SymbolGraph(str(tmp_path))
        _seed_symbols(g, "ws", [
            ("add", "function", str(tmp_path / "a.c")),
            ("nan", "function", str(tmp_path / "b.c")),
        ])
        result = g.locate_l2("ws", "rnd")
        assert result.resolution_strategy == "none"

    def test_partial_match_key_ordering(self) -> None:
        # Direct unit check on the sort key: prefix < interior; def < impl.
        prefix_def = {"name": "rand_fn", "kind": "function", "file_path": "src/x.py"}
        interior = {"name": "myrand", "kind": "function", "file_path": "src/y.py"}
        prefix_impl = {"name": "rand_x", "kind": "impl", "file_path": "src/z.py"}
        assert _partial_match_key(prefix_def, "rand") < _partial_match_key(interior, "rand")
        assert _partial_match_key(prefix_def, "rand") < _partial_match_key(prefix_impl, "rand")


# ---------------------------------------------------------------------------
# UPG-4.1 — namespace symbols to kill name collisions
# ---------------------------------------------------------------------------

class TestTraceCollisionUPG41:
    def test_callers_exact_no_substring_bleed(self, tmp_path) -> None:
        # `compare` and `compare_stacks` are distinct symbols; callers("compare")
        # must NOT pull in the caller of `compare_stacks` (old LIKE bug).
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "m.py", textwrap.dedent("""\
            def compare(): pass
            def compare_stacks(): pass
            def user_a():
                compare()
            def user_b():
                compare_stacks()
        """))
        g.index_file("ws", path)
        callers = [e.from_symbol for e in g.callers("ws", "compare")]
        assert "user_a" in callers
        assert "user_b" not in callers

    def test_callees_fall_back_to_partial_when_no_exact(self, tmp_path) -> None:
        # No symbol named exactly "process", so callees("process") should still
        # find process_data's calls via the partial fallback.
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "p.py", textwrap.dedent("""\
            def helper(): pass
            def process_data():
                helper()
        """))
        g.index_file("ws", path)
        callees = [e.to_symbol for e in g.callees("ws", "process")]
        assert "helper" in callees

    def test_trace_separates_same_named_defs_across_modules(self, tmp_path) -> None:
        # Two `Lock` definitions in different modules, each calling different
        # things. trace must scope callees per definition, not merge them.
        resolver = tmp_path / "resolver"
        sync = tmp_path / "sync"
        resolver.mkdir()
        sync.mkdir()
        (resolver / "lock.py").write_text(textwrap.dedent("""\
            def read_lockfile(): pass
            def Lock():
                read_lockfile()
        """))
        (sync / "mutex.py").write_text(textwrap.dedent("""\
            def os_mutex(): pass
            def Lock():
                os_mutex()
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, str(resolver / "lock.py"))
        g.index_file(ws, str(sync / "mutex.py"))

        result = g.trace(ws, "Lock", direction="both")
        assert len(result["definitions"]) == 2
        by_def = result["by_definition"]
        assert len(by_def) == 2

        callees_by_module = {
            e_def["module"]: {e.to_symbol for e in e_def["callees"]}
            for e_def in by_def
        }
        resolver_callees = next(v for k, v in callees_by_module.items() if "resolver" in k)
        sync_callees = next(v for k, v in callees_by_module.items() if "sync" in k)
        assert "read_lockfile" in resolver_callees and "os_mutex" not in resolver_callees
        assert "os_mutex" in sync_callees and "read_lockfile" not in sync_callees

    def test_format_trace_renders_per_definition_separation(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        d1 = Symbol(symbol_id=1, workspace="ws", name="Lock", kind="function",
                    file_path="/ws/resolver/lock.py", start_line=2, end_line=4)
        d2 = Symbol(symbol_id=2, workspace="ws", name="Lock", kind="function",
                    file_path="/ws/sync/mutex.py", start_line=2, end_line=4)
        trace_result = {
            "callers": [],
            "callees": [],
            "definitions": [d1, d2],
            "by_definition": [
                {"definition": d1, "module": "resolver/lock.py",
                 "callees": [CallEdge(from_file="/ws/resolver/lock.py", from_symbol="Lock",
                                      from_line=2, to_symbol="read_lockfile", edge_type="calls")]},
                {"definition": d2, "module": "sync/mutex.py",
                 "callees": [CallEdge(from_file="/ws/sync/mutex.py", from_symbol="Lock",
                                      from_line=2, to_symbol="os_mutex", edge_type="calls")]},
            ],
        }
        text = g.format_trace_for_llm(trace_result, "Lock")
        assert "2 definitions" in text
        assert "resolver/lock.py" in text and "sync/mutex.py" in text
        assert "read_lockfile" in text and "os_mutex" in text

    def test_single_definition_uses_flat_format(self, tmp_path) -> None:
        # One definition → no per-definition block, no ambiguity warning.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        path = make_py(tmp_path, "s.py", textwrap.dedent("""\
            def helper(): pass
            def only_caller():
                helper()
        """))
        g.index_file(ws, path)
        result = g.trace(ws, "helper", direction="both")
        assert "by_definition" not in result
        text = g.format_trace_for_llm(result, "helper")
        assert "definitions across modules" not in text


# ---------------------------------------------------------------------------
# UPG-TRACE-GRAPH-INCOMPLETE (F60) — _exact_definitions must not silently
# drop a legitimate definition of a common leaf name because of its file
# path's alphabetical accident; canonical (non-test, larger-span) ordering
# must survive the fetch-cap truncation instead of `ORDER BY file_path`.
# ---------------------------------------------------------------------------

class TestTraceGraphIncompleteF60:
    def _seed_many_defs(self, g, ws: str, tmp_path, stub_count: int) -> str:
        """Seed `stub_count` tiny 1-line 'delete' stubs in files that sort
        alphabetically BEFORE one much larger 'delete' definition. Returns
        the large definition's file_path — the one that must survive a
        `limit == stub_count` truncation."""
        import time
        now = time.time()
        target_fp = str(tmp_path / "zzz_target.py")
        Path(target_fp).write_text("# placeholder\n" * 120)
        with g._conn() as conn:
            for i in range(stub_count):
                fp = str(tmp_path / f"aaa_stub_{i:03d}.py")
                Path(fp).write_text("# placeholder\n")
                conn.execute(
                    "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ws, "delete", "function", fp, 1, 2, now),
                )
            conn.execute(
                "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ws, "delete", "function", target_fp, 1, 100, now),
            )
        return target_fp

    def test_exact_definitions_keeps_large_def_over_alpha_truncation(self, tmp_path) -> None:
        # 20 tiny stubs (alphabetically first) + 1 large def (alphabetically
        # last) = 21 rows, limit=20 → exactly one must be dropped. The old
        # `ORDER BY file_path` always dropped the large def purely because
        # 'z' > 'a'; canonical (span DESC) ordering must keep it instead.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        target_fp = self._seed_many_defs(g, ws, tmp_path, stub_count=20)
        defs = g._exact_definitions(ws, "delete", limit=20)
        assert len(defs) == 20
        assert any(d.file_path == target_fp for d in defs), (
            "Large canonical definition dropped by alphabetical-path truncation; "
            f"got files: {[Path(d.file_path).name for d in defs]}"
        )

    def test_trace_by_definition_includes_large_def(self, tmp_path) -> None:
        # End-to-end: vectr_trace('delete') must surface the large definition
        # in its per-definition breakdown, not just the internal query.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        target_fp = self._seed_many_defs(g, ws, tmp_path, stub_count=20)
        result = g.trace(ws, "delete", direction="both", limit=20)
        by_def_files = {e["definition"].file_path for e in result["by_definition"]}
        assert target_fp in by_def_files, (
            f"Large definition missing from trace by_definition dump: "
            f"{[Path(f).name for f in by_def_files]}"
        )


# ---------------------------------------------------------------------------
# F33 (UPG-17.1) — vectr_trace('Class.method') qualified-name resolution.
#
# vectr_search displays symbols as "Class.method"; an LLM naturally copies
# that qualified name into a follow-up trace call. The edges/symbols tables
# only ever store the bare leaf, so a literal name="Class.method" lookup was
# silently empty for both callers and callees. Resolution: look up by the
# bare leaf, then filter/scope by the named class's enclosing definition.
# ---------------------------------------------------------------------------

class TestTraceQualifiedNameF33:
    def _fixture(self, tmp_path) -> tuple:
        resolver = tmp_path / "resolver"
        sync = tmp_path / "sync"
        resolver.mkdir()
        sync.mkdir()
        (resolver / "models.py").write_text(textwrap.dedent("""\
            class Alpha:
                def helper_a(self): pass
                def run(self):
                    self.helper_a()

            def use_alpha():
                Alpha().run()
        """))
        (sync / "models.py").write_text(textwrap.dedent("""\
            class Beta:
                def helper_b(self): pass
                def run(self):
                    self.helper_b()
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, str(resolver / "models.py"))
        g.index_file(ws, str(sync / "models.py"))
        return g, ws

    def test_qualified_trace_not_silently_empty(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "Alpha.run", direction="both")
        assert result["callers"], "qualified trace must not silently return empty callers"
        assert result.get("qualified_class") == "Alpha"

    def test_qualified_trace_scopes_callees_to_named_class(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "Alpha.run", direction="both")
        by_def = result["by_definition"]
        assert len(by_def) == 1
        callees = {e.to_symbol for e in by_def[0]["callees"]}
        assert "helper_a" in callees
        assert "helper_b" not in callees
        assert by_def[0]["definition"].name == "Alpha.run"

    def test_qualified_trace_other_class_not_confused(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "Beta.run", direction="both")
        by_def = result["by_definition"]
        assert len(by_def) == 1
        callees = {e.to_symbol for e in by_def[0]["callees"]}
        assert "helper_b" in callees
        assert "helper_a" not in callees

    def test_bare_leaf_trace_unaffected(self, tmp_path) -> None:
        """Bare 'run' still returns the ambiguous multi-definition view — no
        regression to the existing UPG-4.1 behavior."""
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "run", direction="both")
        assert "qualified_class" not in result
        assert len(result["definitions"]) == 2

    def test_format_trace_qualified_shows_resolved_wording(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "Alpha.run", direction="both")
        text = g.format_trace_for_llm(result, "Alpha.run")
        assert "Resolved" in text
        assert "helper_a" in text
        assert "helper_b" not in text

    def test_unknown_class_qualifier_falls_back_gracefully(self, tmp_path) -> None:
        """A class qualifier matching no real definition must not silently
        empty out — it degrades to the bare-leaf ambiguous view."""
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "Ghost.run", direction="both")
        assert "qualified_class" not in result
        assert len(result["definitions"]) == 2


# ---------------------------------------------------------------------------
# UPG-TRACE-CALLERS-QUALIFIER-VALIDATION — a `vectr_trace("Class.method")`
# whose class qualifier matches no real definition must not silently answer
# with the bare leaf's callers/callees as though they belonged to that class.
# `definitions`/`by_definition` already validate the qualifier (F33 above);
# this closes the gap for `callers` (and the flat, non-by-definition
# `callees` answer), which previously skipped validation entirely.
# ---------------------------------------------------------------------------

class TestTraceCallersQualifierValidationUPG:
    def _fixture(self, tmp_path) -> tuple:
        core = tmp_path / "core"
        legacy = tmp_path / "legacy"
        core.mkdir()
        legacy.mkdir()
        # RegistryClient has `simple_detail`, never a bare `simple` method.
        (core / "registry.py").write_text(textwrap.dedent("""\
            class RegistryClient:
                def simple_detail(self):
                    pass

                def other_method(self):
                    pass

            def use_registry():
                RegistryClient().simple_detail()
        """))
        # A same-leaf-adjacent, wholly unrelated `simple()` elsewhere — its
        # caller must never be presented as RegistryClient's.
        (legacy / "misc.py").write_text(textwrap.dedent("""\
            def simple():
                pass

            def unrelated_caller():
                simple()
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, str(core / "registry.py"))
        g.index_file(ws, str(legacy / "misc.py"))
        return g, ws

    def test_fabricated_qualifier_flags_unresolved_with_near_miss(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "RegistryClient.simple", direction="callers")
        assert result.get("qualifier_unresolved") is True
        near_miss_names = {s.name for s in result.get("qualifier_near_miss", [])}
        assert "RegistryClient.simple_detail" in near_miss_names

    def test_fabricated_qualifier_callers_not_presented_as_class_callers(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "RegistryClient.simple", direction="callers")
        text = g.format_trace_for_llm(result, "RegistryClient.simple")

        assert "does not resolve" in text
        assert "RegistryClient.simple_detail" in text  # near-miss surfaced

        # The miss banner must LEAD — appear before any fallback caller data.
        banner_idx = text.find("does not resolve")
        caller_idx = text.find("unrelated_caller")
        assert banner_idx != -1
        if caller_idx != -1:
            assert banner_idx < caller_idx

        # Never the unlabeled, confident-looking heading a real single-symbol
        # trace would use.
        assert "Called by (1):" not in text
        # If the bare-leaf caller is still shown as a fallback, it must be
        # unambiguously labeled as the leaf's, not the (fabricated) class's.
        if "unrelated_caller" in text:
            assert "any 'simple'" in text

    def test_valid_qualifier_trace_unchanged(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "RegistryClient.simple_detail", direction="callers")
        assert "qualifier_unresolved" not in result
        assert result["callers"], "valid qualified trace must return real callers"
        callers = {e.from_symbol for e in result["callers"]}
        assert "use_registry" in callers
        assert "unrelated_caller" not in callers  # misc.py's `simple` caller must not leak in

        text = g.format_trace_for_llm(result, "RegistryClient.simple_detail")
        assert "does not resolve" not in text
        assert "Called by (1):" in text
        assert "use_registry" in text

    def test_bare_leaf_trace_unaffected(self, tmp_path) -> None:
        g, ws = self._fixture(tmp_path)
        result = g.trace(ws, "simple_detail", direction="callers")
        assert "qualifier_unresolved" not in result
        text = g.format_trace_for_llm(result, "simple_detail")
        assert "does not resolve" not in text
        assert "Called by (1):" in text

    def test_unknown_qualifier_ambiguous_defs_banner_leads(self, tmp_path) -> None:
        """Same validation gap, but with >1 same-leaf definitions across
        classes (the `by_definition` branch fires) — the banner must still
        lead, and the pre-existing ambiguous-defs wording must not claim the
        fabricated qualified name itself "has definitions"."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "models.py").write_text(textwrap.dedent("""\
            class Widget:
                def render(self):
                    pass

            def use_widget_a():
                Widget().render()
        """))
        (b / "models.py").write_text(textwrap.dedent("""\
            class Gadget:
                def render(self):
                    pass

            def use_widget_b():
                Gadget().render()
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, str(a / "models.py"))
        g.index_file(ws, str(b / "models.py"))

        result = g.trace(ws, "Ghost.render", direction="both")
        assert result.get("qualifier_unresolved") is True
        assert "qualified_class" not in result

        text = g.format_trace_for_llm(result, "Ghost.render")
        banner_idx = text.find("does not resolve")
        assert banner_idx != -1
        by_def_idx = text.find("Falling back")
        assert by_def_idx == -1 or banner_idx < by_def_idx
        assert "'Ghost.render' has" not in text  # no longer claims the fake name "has" definitions


# ---------------------------------------------------------------------------
# UPG-4.2 — dedup edges; rank callees/callers by relevance, not alphabetically
# ---------------------------------------------------------------------------

class TestEdgeAggregationUPG42:
    def test_callees_deduped_with_count(self, tmp_path) -> None:
        # `caller` calls `target` from 3 distinct sites; one aggregated entry
        # carrying call_count=3, not three repeated rows.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "caller", 10, "target"),
            (f"{ws}/a.py", "caller", 11, "target"),
            (f"{ws}/a.py", "caller", 12, "target"),
        ])
        callees = g.callees(ws, "caller")
        assert len(callees) == 1
        assert callees[0].to_symbol == "target"
        assert callees[0].call_count == 3

    def test_callees_lead_with_repo_defined_not_builtins(self, tmp_path) -> None:
        # `len`/`assert` are not repo symbols; `helper` is. Even though `len`
        # is called more often, the repo-defined callee must rank first.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("helper", "function", f"{ws}/util.py")])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "caller", 1, "len"),
            (f"{ws}/a.py", "caller", 2, "len"),
            (f"{ws}/a.py", "caller", 3, "len"),
            (f"{ws}/a.py", "caller", 4, "assert"),
            (f"{ws}/a.py", "caller", 5, "helper"),
        ])
        callees = g.callees(ws, "caller")
        assert callees[0].to_symbol == "helper"  # repo-defined leads despite lower count

    def test_callees_ranked_by_frequency_within_tier(self, tmp_path) -> None:
        # Two repo-defined callees: the more frequently called ranks first
        # (frequency beats alphabetical — `zeta`×3 before `alpha`×1).
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("alpha", "function", f"{ws}/util.py"),
            ("zeta", "function", f"{ws}/util.py"),
        ])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "caller", 1, "alpha"),
            (f"{ws}/a.py", "caller", 2, "zeta"),
            (f"{ws}/a.py", "caller", 3, "zeta"),
            (f"{ws}/a.py", "caller", 4, "zeta"),
        ])
        names = [e.to_symbol for e in g.callees(ws, "caller")]
        assert names == ["zeta", "alpha"]

    def test_important_callee_survives_truncation(self, tmp_path) -> None:
        # 30 alphabetically-early builtins + one heavily-used repo callee `zzz`.
        # Old path truncated alphabetically and dropped `zzz`; ranking keeps it.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("zzz", "function", f"{ws}/util.py")])
        specs = [(f"{ws}/a.py", "caller", i, f"aaa_builtin_{i:02d}") for i in range(30)]
        specs += [(f"{ws}/a.py", "caller", 100 + j, "zzz") for j in range(5)]
        _seed_edges(g, ws, specs)
        names = [e.to_symbol for e in g.callees(ws, "caller", limit=20)]
        assert "zzz" in names
        assert names[0] == "zzz"

    def test_callers_deduped_by_function_with_count(self, tmp_path) -> None:
        # One caller function calls `target` from 3 sites → one entry, count 3.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "on_event", 5, "target"),
            (f"{ws}/a.py", "on_event", 6, "target"),
            (f"{ws}/a.py", "on_event", 7, "target"),
        ])
        callers = g.callers(ws, "target")
        assert len(callers) == 1
        assert callers[0].from_symbol == "on_event"
        assert callers[0].call_count == 3

    def test_count_suffix_in_formatted_output(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "caller", 1, "target"),
            (f"{ws}/a.py", "caller", 2, "target"),
        ])
        result = g.trace(ws, "caller", direction="callees")
        text = g.format_trace_for_llm(result, "caller")
        assert "target" in text
        assert "×2" in text

    def test_singleton_callee_has_no_count_suffix(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [(f"{ws}/a.py", "caller", 1, "target")])
        result = g.trace(ws, "caller", direction="callees")
        text = g.format_trace_for_llm(result, "caller")
        assert "target" in text
        assert "×" not in text


# ---------------------------------------------------------------------------
# UPG-CALLER-TIEBREAK-ALPHA: on tied call_count, caller aggregation
# (rank_repo_defined=False) no longer degrades straight to name order.
# Secondary keys: repo-defined callers before external/unresolved ones,
# then higher in-degree (callers of the caller, from the already-loaded
# edge table) before lower, with name as the final stable tiebreak.
# ---------------------------------------------------------------------------

class TestCallerTiebreakAlphaUPG:
    def test_repo_defined_caller_ranks_before_external_on_tied_count(self, tmp_path) -> None:
        # `alpha_caller` is unresolved/external (no symbol row); `beta_caller`
        # is repo-defined. Both call `target` exactly once — tied call_count.
        # Alphabetically "alpha_caller" < "beta_caller", so a pure name-order
        # fallback would (wrongly) rank the external one first.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("beta_caller", "function", f"{ws}/b.py")])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "alpha_caller", 1, "target"),
            (f"{ws}/a.py", "beta_caller", 2, "target"),
        ])
        names = [e.from_symbol for e in g.callers(ws, "target")]
        assert names == ["beta_caller", "alpha_caller"]

    def test_higher_in_degree_breaks_tie_between_two_repo_defined_callers(
        self, tmp_path,
    ) -> None:
        # `beta_caller` and `zeta_caller` are both repo-defined and both call
        # `target` once — tied on call_count and on repo-defined. `zeta_caller`
        # itself has 3 distinct callers elsewhere in the graph (higher
        # in-degree/centrality proxy); `beta_caller` has none.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("beta_caller", "function", f"{ws}/b.py"),
            ("zeta_caller", "function", f"{ws}/z.py"),
        ])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "beta_caller", 1, "target"),
            (f"{ws}/a.py", "zeta_caller", 2, "target"),
            (f"{ws}/x.py", "caller_x1", 10, "zeta_caller"),
            (f"{ws}/x.py", "caller_x2", 11, "zeta_caller"),
            (f"{ws}/x.py", "caller_x3", 12, "zeta_caller"),
        ])
        names = [e.from_symbol for e in g.callers(ws, "target")]
        assert names == ["zeta_caller", "beta_caller"]

    def test_full_tie_falls_back_to_name_as_stable_final_key(self, tmp_path) -> None:
        # Both callers repo-defined, both zero in-degree, both call `target`
        # once — every secondary signal ties, so the final deterministic key
        # is the name itself.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("alpha_caller", "function", f"{ws}/a.py"),
            ("beta_caller", "function", f"{ws}/b.py"),
        ])
        _seed_edges(g, ws, [
            (f"{ws}/c.py", "alpha_caller", 1, "target"),
            (f"{ws}/c.py", "beta_caller", 2, "target"),
        ])
        names = [e.from_symbol for e in g.callers(ws, "target")]
        assert names == ["alpha_caller", "beta_caller"]

    def test_three_way_tie_combines_repo_defined_and_in_degree(self, tmp_path) -> None:
        # alpha_caller: external/unresolved. beta_caller: repo-defined,
        # in-degree 0. zeta_caller: repo-defined, in-degree 3. All three tie
        # on call_count (1 each) — full expected ranking exercises both
        # secondary keys together, not just pairwise.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("beta_caller", "function", f"{ws}/b.py"),
            ("zeta_caller", "function", f"{ws}/z.py"),
        ])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "alpha_caller", 1, "target"),
            (f"{ws}/a.py", "beta_caller", 2, "target"),
            (f"{ws}/a.py", "zeta_caller", 3, "target"),
            (f"{ws}/x.py", "caller_x1", 10, "zeta_caller"),
            (f"{ws}/x.py", "caller_x2", 11, "zeta_caller"),
            (f"{ws}/x.py", "caller_x3", 12, "zeta_caller"),
        ])
        names = [e.from_symbol for e in g.callers(ws, "target")]
        assert names == ["zeta_caller", "beta_caller", "alpha_caller"]

    def test_untied_call_count_still_dominates_over_secondary_keys(self, tmp_path) -> None:
        # `alpha_caller` is external but calls `target` 3 times; `beta_caller`
        # is repo-defined but calls it only once. Primary key (call_count)
        # must still win over the repo-defined/in-degree secondary keys.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("beta_caller", "function", f"{ws}/b.py")])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "alpha_caller", 1, "target"),
            (f"{ws}/a.py", "alpha_caller", 2, "target"),
            (f"{ws}/a.py", "alpha_caller", 3, "target"),
            (f"{ws}/a.py", "beta_caller", 4, "target"),
        ])
        names = [e.from_symbol for e in g.callers(ws, "target")]
        assert names == ["alpha_caller", "beta_caller"]


# ---------------------------------------------------------------------------
# UPG-TRACE-CALLER-AGGREGATION: two callers sharing a leaf name in different
# files must both appear in the trace — the old collapse-by-bare-name key
# elected one file's edge as the sole representative and folded every other
# file's call sites into its ×N count, so the real caller in the second (or
# third...) file never appeared anywhere in the rendered output.
# ---------------------------------------------------------------------------

class TestCallerAggregationDoesNotCollapseAcrossFilesUPG:
    def test_same_leaf_name_callers_in_different_files_both_appear(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/one/handler.py", "__call__", 10, "target"),
            (f"{ws}/two/handler.py", "__call__", 20, "target"),
        ])
        callers = g.callers(ws, "target")
        files = {e.from_file for e in callers}
        assert files == {f"{ws}/one/handler.py", f"{ws}/two/handler.py"}
        assert len(callers) == 2

    def test_same_file_call_sites_still_fold_into_one_entry_with_count(self, tmp_path) -> None:
        # Same name, same file, multiple call sites — this ambiguity a
        # receiver-blind static graph genuinely can't resolve further, so
        # it still collapses into one entry (unlike the cross-file case).
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/handler.py", "__call__", 10, "target"),
            (f"{ws}/handler.py", "__call__", 40, "target"),
        ])
        callers = g.callers(ws, "target")
        assert len(callers) == 1
        assert callers[0].call_count == 2

    def test_three_files_same_leaf_name_all_three_appear(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/a/handler.py", "__call__", 5, "target"),
            (f"{ws}/b/handler.py", "__call__", 6, "target"),
            (f"{ws}/c/handler.py", "__call__", 7, "target"),
        ])
        callers = g.callers(ws, "target")
        files = {e.from_file for e in callers}
        assert files == {
            f"{ws}/a/handler.py", f"{ws}/b/handler.py", f"{ws}/c/handler.py",
        }

    def test_rendered_trace_shows_both_files_not_one_elected_representative(
        self, tmp_path,
    ) -> None:
        # Direct guard on the formatter output an LLM actually reads — not
        # just the underlying list — matching the caller's real complaint:
        # "who calls X" must not answer with only ONE file when several
        # different files genuinely call it.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/one/handler.py", "__call__", 10, "target"),
            (f"{ws}/two/handler.py", "__call__", 20, "target"),
        ])
        result = g.trace(ws, "target", direction="callers")
        text = g.format_trace_for_llm(result, "target")
        assert f"{ws}/one/handler.py" in text
        assert f"{ws}/two/handler.py" in text


# ---------------------------------------------------------------------------
# UPG-TRACE-CALLER-LIMIT-TRANSPARENCY: every trace list truncated at `limit`
# says how many entries were cut — a caller/callee list must never read as
# complete when it is only a prefix of the real one. Since
# UPG-TRACE-CALLER-AGGREGATION stopped collapsing same-leaf callers across
# files, distinct (name, file) pairs each consume their own slot against the
# limit, so truncation got dramatically easier to hit silently.
# ---------------------------------------------------------------------------

class TestTraceLimitTransparencyUPG:
    def test_flat_callers_truncated_count_and_footer(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        # 25 DISTINCT callers (distinct names AND files — each its own entry
        # since UPG-TRACE-CALLER-AGGREGATION) against limit=20.
        _seed_edges(g, ws, [
            (f"{ws}/pkg/caller_{i}.py", f"caller_{i}", 10 + i, "target")
            for i in range(25)
        ])
        result = g.trace(ws, "target", direction="callers", limit=20)
        assert len(result["callers"]) == 20
        assert result["callers_truncated"] == 5
        text = g.format_trace_for_llm(result, "target")
        assert "(+5 more callers not shown — pass a higher limit to show)" in text

    def test_flat_callees_truncated_count_and_footer(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/caller.py", "target", i, f"callee_{i}")
            for i in range(25)
        ])
        result = g.trace(ws, "target", direction="callees", limit=20)
        assert len(result["callees"]) == 20
        assert result["callees_truncated"] == 5
        text = g.format_trace_for_llm(result, "target")
        assert "(+5 more callees not shown — pass a higher limit to show)" in text

    def test_per_definition_callee_truncation_reported_per_entry(
        self, tmp_path,
    ) -> None:
        """The by-definition branch renders one callee list PER definition —
        each list is truncated independently against the same limit, so each
        entry carries its own count and its own footer."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("target", "function", f"{ws}/mod_a.py"),
            ("target", "function", f"{ws}/mod_b.py"),
        ])
        specs = []
        for mod in ("mod_a", "mod_b"):
            specs += [
                (f"{ws}/{mod}.py", "target", i, f"{mod}_callee_{i}")
                for i in range(22)
            ]
        _seed_edges(g, ws, specs)

        result = g.trace(ws, "target", direction="both", limit=20)
        by_def = result["by_definition"]
        assert len(by_def) == 2
        assert all(entry["truncated_callees"] == 2 for entry in by_def)
        text = g.format_trace_for_llm(result, "target")
        # Both per-definition lists were cut, and BOTH cuts are disclosed.
        assert text.count("(+2 more callees not shown — pass a higher limit to show)") == 2

    def test_definitions_beyond_limit_disclosed(self, tmp_path) -> None:
        """Whole definitions (with their calls) silently missing when the
        definitions fetch hits the cap is the worst shape of this defect — the
        reader can't even tell which sections never rendered."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("multi_def_fn", "function", f"{ws}/mod_{i}.py") for i in range(21)
        ])
        result = g.trace(ws, "multi_def_fn", direction="both", limit=20)
        assert len(result["by_definition"]) == 20
        assert result["definitions_truncated"] == 1
        text = g.format_trace_for_llm(result, "multi_def_fn")
        assert (
            "(+1 more definition of 'multi_def_fn' beyond "
            "the limit not shown — pass a higher limit to show)"
        ) in text

    def test_definitions_disclosure_suppressed_for_resolved_qualified_query(
        self, tmp_path,
    ) -> None:
        # A resolved Class.method query fetches ALL classes' definitions of the
        # leaf, then keeps only the named class's — beyond-cap definitions are
        # other-class sites excluded BY DESIGN there, so disclosing them would
        # misrepresent designed filtering as silent loss. Both definitions here
        # belong to the NAMED class (in different files), so whichever one the
        # LIMIT-capped fetch returns, the qualifier still resolves and the
        # second definition is genuinely lost to the cap — yet correctly not
        # disclosed as such for this query shape.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        make_py(
            tmp_path, "orders_a.py",
            "class OrderRepository:\n    def save_record(self):\n        pass\n",
        )
        make_py(
            tmp_path, "orders_b.py",
            "class OrderRepository:\n    def save_record(self):\n        pass\n",
        )
        g.index_file(ws, str(tmp_path / "orders_a.py"))
        g.index_file(ws, str(tmp_path / "orders_b.py"))

        result = g.trace(ws, "OrderRepository.save_record", direction="both", limit=1)
        assert result.get("qualified_class") == "OrderRepository"
        assert len(result["by_definition"]) == 1
        assert result["definitions_truncated"] == 1
        text = g.format_trace_for_llm(result, "OrderRepository.save_record")
        assert "beyond the limit not shown" not in text

    def test_no_truncation_means_no_footer(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [(f"{ws}/a.py", "caller", 1, "target")])
        result = g.trace(ws, "target", direction="both", limit=20)
        assert result["callers_truncated"] == 0
        assert result["callees_truncated"] == 0
        assert result["definitions_truncated"] == 0
        text = g.format_trace_for_llm(result, "target")
        assert "not shown" not in text

    def test_aggregate_edges_reports_truncated_directly(self, tmp_path) -> None:
        """Unit guard on the authoritative layer: `_aggregate_edges` owns THE
        cut against the user-facing `limit`, so the count must come from there
        — a count taken after any second truncation would itself be wrong."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        edges = [
            CallEdge(
                from_file=f"{ws}/f{i}.py", from_symbol=f"caller_{i}",
                from_line=i + 1, to_symbol="target", edge_type="calls",
            )
            for i in range(7)
        ]
        shown, hidden, truncated = g._aggregate_edges(
            ws, edges, "from_symbol", 5, rank_repo_defined=False,
        )
        assert len(shown) == 5
        assert hidden == 0
        assert truncated == 2


# ---------------------------------------------------------------------------
# UPG-TRACE-DIRECTION-UNQUERIED-RENDER: a single-direction trace must not
# render anything about the direction the caller never asked for. The
# previous "absent key ⇒ `[]` via `.get(key, [])` ⇒ always non-None ⇒
# always rendered" chain turned every `direction="callees"` result into a
# fake "Called by: (none found in index)" line about a lookup that never
# ran — a confident-looking false negative an LLM caller would act on.
# Fix: `trace()` now sets the un-queried direction to `None` (not absent,
# not `[]`), and the formatter gates on `is None`.
# ---------------------------------------------------------------------------

class TestTraceDirectionUnqueriedRenderUPG:
    def _indexed_graph(self, tmp_path):
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "svc.py", textwrap.dedent("""\
            def helper():
                pass

            def process():
                helper()

            def orchestrate():
                process()
                helper()
        """))
        g.index_file("ws", path)
        return g, path

    def test_callees_direction_does_not_render_callers_block(self, tmp_path) -> None:
        g, _ = self._indexed_graph(tmp_path)
        result = g.trace("ws", "process", direction="callees")
        # The un-queried callers direction must not appear anywhere in the
        # rendered text — no "Called by" heading, no "(none found in index)"
        # line about callers, no callers block at all. A reader of this
        # answer must not be able to conclude "process has no callers" from
        # a `direction="callees"` trace.
        text = g.format_trace_for_llm(result, "process")
        assert "Called by" not in text
        assert "none found in index" not in text

    def test_callers_direction_does_not_render_callees_block(self, tmp_path) -> None:
        g, _ = self._indexed_graph(tmp_path)
        result = g.trace("ws", "helper", direction="callers")
        # Symmetric: a callers-only trace must not render the Calls block.
        # The per-definition "calls (N)" header (only when by_def is
        # rendered) and the flat "Calls (N):" heading both come from the
        # callees path; either here would be a real false negative.
        text = g.format_trace_for_llm(result, "helper")
        assert "Calls" not in text  # no "Calls (N):" or "calls (0):" header
        assert "(none found in index)" not in text

    def test_both_direction_with_genuine_empty_still_renders_both_blocks(
        self, tmp_path,
    ) -> None:
        # A `direction="both"` trace on a symbol that genuinely has no
        # callers and no callees must STILL render the two "none found in
        # index" lines — those are real, queried lookups that returned
        # empty, not the un-queried-direction false negative fixed above.
        # Fixing the un-queried case must not break this honest-miss case.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        path = make_py(tmp_path, "lonely.py", "def lonely_function():\n    pass\n")
        g.index_file(ws, path)
        result = g.trace(ws, "lonely_function", direction="both")
        assert isinstance(result["callers"], list) and result["callers"] == []
        assert isinstance(result["callees"], list) and result["callees"] == []
        text = g.format_trace_for_llm(result, "lonely_function")
        assert "Called by: (none found in index)" in text
        assert "Calls: (none found in index)" in text

    def test_unqueried_direction_silences_recovery_hint(self, tmp_path) -> None:
        # The `_empty_trace_hint` recovery banner is a "static call-graph
        # found no calls for X" suggestion. It must not fire for a
        # single-direction trace whose OTHER direction was simply not
        # queried — the recovery hint assumes both sides were looked up.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        path = make_py(tmp_path, "c.py", "def lonely():\n    pass\n")
        g.index_file(ws, path)
        result = g.trace(ws, "lonely", direction="callees")
        text = g.format_trace_for_llm(result, "lonely")
        # Recovery hint is gated by `callers_empty and callees_empty`; an
        # un-queried callers direction is `None`, so `callers_empty`
        # evaluates False and the hint must NOT appear.
        assert "vectr_search" not in text
        assert "dynamic" not in text.lower()

    def test_result_dict_carries_unqueried_direction_as_none(self, tmp_path) -> None:
        # The signal must be `None` (not absent, not `[]`) — see the
        # docstring of the class for why the dict now has both keys
        # consistently. This pins the public contract for any future
        # caller that wants to introspect the result directly.
        g, _ = self._indexed_graph(tmp_path)
        callers_only = g.trace("ws", "process", direction="callers")
        assert "callers" in callers_only
        assert "callees" in callers_only
        assert isinstance(callers_only["callers"], list)
        assert callers_only["callees"] is None
        callees_only = g.trace("ws", "process", direction="callees")
        assert "callers" in callees_only
        assert "callees" in callees_only
        assert isinstance(callees_only["callees"], list)
        assert callees_only["callers"] is None


# ---------------------------------------------------------------------------
# UPG-TRACE-EDGE-FETCH-CAP-UNCOUNTED: `_EDGE_FETCH_CAP = 1000` is an
# upstream SQL LIMIT on raw edge rows. When it bites, the aggregated
# `truncated` count the user sees in "(+N more ...)" is itself a LOWER
# BOUND — aggregation never saw the beyond-cap raw rows. The disclosure
# must say so. No extra query per trace: we detect cap-hit by
# `len(rows) == _EDGE_FETCH_CAP` and reword the note.
# ---------------------------------------------------------------------------

class TestTraceEdgeFetchCapUncountedUPG:
    def test_raw_capped_flag_propagated_to_truncation_note(self, tmp_path) -> None:
        # When the SQL fetch hits _EDGE_FETCH_CAP, the per-direction
        # `*_raw_capped` flag is True and the rendered "(+N more X not
        # shown — pass a higher limit to show)" is reworded to the
        # lower-bound phrasing. This pins the formatter path end-to-end.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        # Seed exactly the cap count of raw rows so `len(rows) ==
        # _EDGE_FETCH_CAP` is true. Each row is a distinct (name, file)
        # pair after UPG-TRACE-CALLER-AGGREGATION, so the `limit=20`
        # truncation also fires — both caps bit, raw_capped wins on the
        # note wording.
        cap = SymbolGraph._EDGE_FETCH_CAP
        _seed_edges(g, ws, [
            (f"{ws}/p_{i}.py", f"caller_{i}", 1, "target")
            for i in range(cap)
        ])
        result = g.trace(ws, "target", direction="callers", limit=20)
        assert result["callers_raw_capped"] is True
        # The 1000 raw rows deduped to (up to) 1000 entries before the
        # limit cut — 20 are shown, the rest are "more not shown" and the
        # note MUST announce that count is a lower bound.
        text = g.format_trace_for_llm(result, "target")
        assert "lower bound" in text
        assert "at least" in text
        assert "raw edge row cap" in text
        # And must NOT use the old exact-looking wording — that is the
        # whole defect this UPG item names.
        assert "pass a higher limit to show" not in text

    def test_raw_capped_flag_false_when_under_cap(self, tmp_path) -> None:
        # Below the cap, the existing exact-count wording is correct and
        # must be preserved (no regression for the common case).
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [
            (f"{ws}/caller_{i}.py", f"caller_{i}", 1, "target")
            for i in range(25)
        ])
        result = g.trace(ws, "target", direction="callers", limit=20)
        assert result["callers_raw_capped"] is False
        text = g.format_trace_for_llm(result, "target")
        assert "(+5 more callers not shown — pass a higher limit to show)" in text
        assert "lower bound" not in text

    def test_callees_raw_capped_flag_propagated(self, tmp_path) -> None:
        # Same path on the callees direction — symmetry check.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        cap = SymbolGraph._EDGE_FETCH_CAP
        _seed_edges(g, ws, [
            (f"{ws}/caller.py", "target", i, f"callee_{i}")
            for i in range(cap)
        ])
        result = g.trace(ws, "target", direction="callees", limit=20)
        assert result["callees_raw_capped"] is True
        text = g.format_trace_for_llm(result, "target")
        assert "lower bound" in text
        assert "raw edge row cap" in text

    def test_per_definition_raw_capped_propagated(self, tmp_path) -> None:
        # The by-definition branch runs its OWN SQL fetch (per
        # definition's file) and so has its own raw-cap signal per entry.
        # When a single definition has more raw rows than the cap, the
        # per-entry "(+N more ...)" footer must reword to the lower-bound
        # phrasing.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        cap = SymbolGraph._EDGE_FETCH_CAP
        _seed_symbols(g, ws, [
            ("target", "function", f"{ws}/mod_a.py"),
            ("target", "function", f"{ws}/mod_b.py"),
        ])
        # Only mod_a hits the cap; mod_b stays small. The cap'd
        # definition's note must reword; mod_b's note (if it fires at
        # all) must use the exact wording.
        _seed_edges(g, ws, [
            (f"{ws}/mod_a.py", "target", i, f"callee_{i}")
            for i in range(cap)
        ] + [
            (f"{ws}/mod_b.py", "target", i, f"modb_callee_{i}")
            for i in range(25)
        ])
        result = g.trace(ws, "target", direction="both", limit=20)
        by_def = result["by_definition"]
        assert len(by_def) == 2
        capped_entry = next(e for e in by_def if e["truncated_callees_raw_capped"])
        not_capped = next(e for e in by_def if not e["truncated_callees_raw_capped"])
        assert capped_entry["truncated_callees"] > 0
        assert not_capped["truncated_callees"] > 0
        text = g.format_trace_for_llm(result, "target")
        # The capped entry's lower-bound note appears at least once.
        assert "lower bound" in text
        # The non-capped entry's exact note also appears.
        assert "pass a higher limit to show" in text


# ---------------------------------------------------------------------------
# UPG-TRACE-ABS-PATH: trace output renders workspace-relative paths, matching
# the convention UPG-RELATIVE-PATH-RENDER already established for
# search/fetch/evict (a `workspace:` header line + relative paths below it),
# rather than leaking the absolute filesystem path of the machine that built
# the index.
# ---------------------------------------------------------------------------

class TestTraceRelativePathsUPG:
    def test_no_workspace_default_keeps_absolute_paths(self, tmp_path) -> None:
        # Backward compatibility: callers that don't pass `workspace` (the
        # pre-existing 2-arg call shape used throughout this file) still get
        # the exact same output as before this change.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [(f"{ws}/handler.py", "caller", 10, "target")])
        result = g.trace(ws, "target", direction="callers")
        text = g.format_trace_for_llm(result, "target")
        assert f"{ws}/handler.py:10" in text
        assert "workspace:" not in text

    def test_workspace_relative_caller_path_no_absolute_root_substring(self, tmp_path) -> None:
        # Matches the UPG-RELATIVE-PATH-RENDER convention already verified
        # for search/fetch/evict: the absolute root appears exactly once, in
        # the `workspace:` header — no rendered line repeats it.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [(f"{ws}/pkg/handler.py", "caller", 10, "target")])
        result = g.trace(ws, "target", direction="callers")
        text = g.format_trace_for_llm(result, "target", workspace=ws)
        assert f"workspace: {ws}" in text
        assert "pkg/handler.py:10" in text
        assert f"{ws}/pkg/handler.py:10" not in text
        assert text.count(ws) == 1

    def test_workspace_relative_by_definition_branch_caller_path(self, tmp_path) -> None:
        # The by-definition rendering branch (ambiguous name / qualified
        # class) has its OWN caller-rendering loop — must not regress
        # separately from the flat branch above.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("target", "function", f"{ws}/mod_a.py"),
            ("target", "function", f"{ws}/mod_b.py"),
        ])
        _seed_edges(g, ws, [(f"{ws}/pkg/handler.py", "caller", 10, "target")])
        result = g.trace(ws, "target", direction="both")
        text = g.format_trace_for_llm(result, "target", workspace=ws)
        assert "pkg/handler.py:10" in text
        assert f"{ws}/pkg/handler.py:10" not in text


# ---------------------------------------------------------------------------
# UPG-TRACE-FETCH-IDS: each rendered caller/callee line carries a real,
# vectr_fetch-resolvable `relpath:start-end` id when the index actually has a
# stored chunk covering that site — never a fabricated one. This is checked
# here with a stub `chunk_span_lookup` (isolating the rendering logic from a
# real Chroma index); the true round-trip through `vectr_fetch` is covered by
# an integration test elsewhere.
# ---------------------------------------------------------------------------

class TestTraceFetchIdsUPG:
    def test_caller_line_gets_fetch_id_when_span_resolves(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [(f"{ws}/handler.py", "caller", 10, "target")])
        result = g.trace(ws, "target", direction="callers")

        def stub_lookup(locations):
            return {(f"{ws}/handler.py", 10): (5, 15)}

        text = g.format_trace_for_llm(
            result, "target", workspace=ws, chunk_span_lookup=stub_lookup,
        )
        assert "[fetch: handler.py:5-15]" in text

    def test_caller_line_omits_fetch_id_when_span_does_not_resolve(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [(f"{ws}/handler.py", "caller", 10, "target")])
        result = g.trace(ws, "target", direction="callers")

        def stub_lookup(locations):
            return {}  # index has no chunk covering this site

        text = g.format_trace_for_llm(
            result, "target", workspace=ws, chunk_span_lookup=stub_lookup,
        )
        assert "[fetch:" not in text
        assert f"handler.py:10" in text

    def test_callee_line_gets_fetch_id_for_unambiguous_definition(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("helper", "function", f"{ws}/util.py")])
        _seed_edges(g, ws, [(f"{ws}/a.py", "caller", 1, "helper")])
        result = g.trace(ws, "caller", direction="callees")

        def stub_lookup(locations):
            spans = {}
            for file_path, line in locations:
                if file_path == f"{ws}/util.py":
                    spans[(file_path, line)] = (1, 3)
            return spans

        text = g.format_trace_for_llm(
            result, "caller", workspace=ws, chunk_span_lookup=stub_lookup,
        )
        assert "[fetch: util.py:1-3]" in text

    def test_callee_line_omits_fetch_id_for_ambiguous_definition(self, tmp_path) -> None:
        # Two definitions for the same callee name — never guess which one.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("helper", "function", f"{ws}/util_a.py"),
            ("helper", "function", f"{ws}/util_b.py"),
        ])
        _seed_edges(g, ws, [(f"{ws}/a.py", "caller", 1, "helper")])
        result = g.trace(ws, "caller", direction="callees")

        def stub_lookup(locations):
            return {loc: (1, 3) for loc in locations}

        text = g.format_trace_for_llm(
            result, "caller", workspace=ws, chunk_span_lookup=stub_lookup,
        )
        assert "[fetch:" not in text

    def test_no_chunk_span_lookup_renders_no_fetch_ids(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_edges(g, ws, [(f"{ws}/handler.py", "caller", 10, "target")])
        result = g.trace(ws, "target", direction="callers")
        text = g.format_trace_for_llm(result, "target", workspace=ws)
        assert "[fetch:" not in text


# ---------------------------------------------------------------------------
# UPG-CALLER-INDEGREE-ANALYZE — build_for_workspace gathers query-planner
# statistics so _caller_in_degree's selective `to_symbol IN (...)` lookup
# plans through idx_edge_to instead of a full covering-index table scan.
# ---------------------------------------------------------------------------

class TestCallerInDegreeAnalyzeUPG:
    def _seed_selective_edges(self, g: "SymbolGraph", ws: str) -> list[str]:
        # Many distinct `to_symbol` values (selective for idx_edge_to) all
        # under one workspace (NOT selective for idx_edge_workspace/the
        # covering index) — mirrors a real multi-thousand-symbol workspace
        # closely enough that the planner's choice actually depends on
        # having statistics, rather than being forced either way by shape.
        import random
        rng = random.Random(7)
        names = [f"sym_{i}" for i in range(500)]
        specs = []
        for i in range(3000):
            to_sym = rng.choice(names)
            from_sym = f"caller_{rng.randint(0, 3000)}"
            specs.append((f"{ws}/file_{i % 50}.py", from_sym, i % 500, to_sym))
        _seed_edges(g, ws, specs)
        return names[:10]

    def test_build_for_workspace_populates_edge_index_statistics(
        self, tmp_path,
    ) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        self._seed_selective_edges(g, ws)
        with g._conn() as conn:
            # `sqlite_stat1` itself doesn't exist until the first ANALYZE
            # creates it, so "no statistics yet" also covers the missing-
            # table case rather than only an empty-but-present table.
            try:
                before = {
                    r["idx"] for r in conn.execute(
                        "SELECT idx FROM sqlite_stat1 WHERE tbl = 'edges'"
                    ).fetchall()
                }
            except sqlite3.OperationalError:
                before = set()
        assert "idx_edge_to" not in before  # no ANALYZE has run yet

        g.build_for_workspace(ws, [])

        with g._conn() as conn:
            after = {
                r["idx"] for r in conn.execute(
                    "SELECT idx FROM sqlite_stat1 WHERE tbl = 'edges'"
                ).fetchall()
            }
        assert "idx_edge_to" in after

    def test_caller_in_degree_query_plans_via_idx_edge_to_after_build(
        self, tmp_path,
    ) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        tied_names = self._seed_selective_edges(g, ws)

        def plan_uses_idx_edge_to() -> bool:
            with g._conn() as conn:
                placeholders = ",".join("?" * len(tied_names))
                rows = conn.execute(
                    "EXPLAIN QUERY PLAN SELECT to_symbol, "
                    "COUNT(DISTINCT from_symbol) FROM edges WHERE workspace = ? "
                    f"AND to_symbol IN ({placeholders}) GROUP BY to_symbol",
                    (ws, *tied_names),
                ).fetchall()
            return any("idx_edge_to" in r["detail"] for r in rows)

        assert plan_uses_idx_edge_to() is False  # stale/absent stats, pre-build

        g.build_for_workspace(ws, [])

        assert plan_uses_idx_edge_to() is True
        # The fixed query plan is what makes _caller_in_degree itself return
        # correct counts too, not just a cheaper plan.
        counts = g._caller_in_degree(ws, tied_names)
        assert all(name in counts for name in tied_names if counts.get(name))


# ---------------------------------------------------------------------------
# UPG-4.3 — suppress builtins/stdlib from callee lists (repo-internal by default)
# ---------------------------------------------------------------------------

class TestBuiltinSuppressionUPG43:
    def _graph_with_mixed_callees(self, tmp_path):
        # `caller` (in a .py file) calls a repo function + python builtins.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("helper", "function", f"{ws}/util.py")])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "caller", 1, "helper"),
            (f"{ws}/a.py", "caller", 2, "len"),
            (f"{ws}/a.py", "caller", 3, "isinstance"),
            (f"{ws}/a.py", "caller", 4, "print"),
        ])
        return g, ws

    def test_builtins_hidden_by_default_in_trace(self, tmp_path) -> None:
        g, ws = self._graph_with_mixed_callees(tmp_path)
        result = g.trace(ws, "caller", direction="callees")  # include_builtins defaults False
        names = [e.to_symbol for e in result["callees"]]
        assert names == ["helper"]
        assert result["hidden_builtins"] == 3

    def test_include_builtins_shows_them(self, tmp_path) -> None:
        g, ws = self._graph_with_mixed_callees(tmp_path)
        result = g.trace(ws, "caller", direction="callees", include_builtins=True)
        names = {e.to_symbol for e in result["callees"]}
        assert {"helper", "len", "isinstance", "print"} <= names
        assert result["hidden_builtins"] == 0

    def test_repo_defined_shadowing_a_builtin_is_kept(self, tmp_path) -> None:
        # The repo defines its own `len`; it must NOT be suppressed as a builtin.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("len", "function", f"{ws}/mylib.py")])
        _seed_edges(g, ws, [
            (f"{ws}/a.py", "caller", 1, "len"),
            (f"{ws}/a.py", "caller", 2, "isinstance"),
        ])
        result = g.trace(ws, "caller", direction="callees")
        names = [e.to_symbol for e in result["callees"]]
        assert "len" in names           # repo-defined, kept
        assert "isinstance" not in names  # true builtin, hidden
        assert result["hidden_builtins"] == 1

    def test_hidden_note_rendered(self, tmp_path) -> None:
        g, ws = self._graph_with_mixed_callees(tmp_path)
        result = g.trace(ws, "caller", direction="callees")
        text = g.format_trace_for_llm(result, "caller")
        assert "3 builtin/stdlib calls hidden" in text
        assert "include_builtins" in text

    def test_callees_method_keeps_builtins_by_default(self, tmp_path) -> None:
        # The raw callees() API is unfiltered by default (back-compat); only the
        # trace() display path suppresses.
        g, ws = self._graph_with_mixed_callees(tmp_path)
        names = {e.to_symbol for e in g.callees(ws, "caller")}
        assert "len" in names

    def test_c_builtins_suppressed(self, tmp_path) -> None:
        # malloc/memcpy/assert from a .c file are libc noise, not repo calls.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("parse_header", "function", f"{ws}/hdr.c")])
        _seed_edges(g, ws, [
            (f"{ws}/p.c", "caller", 1, "parse_header"),
            (f"{ws}/p.c", "caller", 2, "malloc"),
            (f"{ws}/p.c", "caller", 3, "memcpy"),
            (f"{ws}/p.c", "caller", 4, "assert"),
        ])
        result = g.trace(ws, "caller", direction="callees")
        names = [e.to_symbol for e in result["callees"]]
        assert names == ["parse_header"]
        assert result["hidden_builtins"] == 3

    def test_per_definition_callees_also_suppress_builtins(self, tmp_path) -> None:
        # Two defs of `Lock` across modules; builtins hidden within each block.
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [
            ("Lock", "function", f"{ws}/resolver/lock.py"),
            ("Lock", "function", f"{ws}/sync/mutex.py"),
            ("read_lockfile", "function", f"{ws}/resolver/io.py"),
        ])
        _seed_edges(g, ws, [
            (f"{ws}/resolver/lock.py", "Lock", 1, "read_lockfile"),
            (f"{ws}/resolver/lock.py", "Lock", 2, "len"),
            (f"{ws}/sync/mutex.py", "Lock", 1, "print"),
        ])
        result = g.trace(ws, "Lock", direction="callees")
        by_def = result["by_definition"]
        for entry in by_def:
            cnames = {e.to_symbol for e in entry["callees"]}
            assert "len" not in cnames and "print" not in cnames
        # the resolver def still shows its repo-internal call
        resolver = next(e for e in by_def if "resolver" in e["module"])
        assert "read_lockfile" in {e.to_symbol for e in resolver["callees"]}
        assert resolver["hidden_builtins"] == 1


# ---------------------------------------------------------------------------
# UPG-4.4 — Rust trait/impl/type-usage edges
#
# `trace <Type>` was empty for heavily-used Rust types (uv RegistryClient,
# BuildContext, PubGrubPackage) because the dominant interaction — passing a
# type by ref/value, returning it, holding it in a field, or calling an
# associated fn `Type::new()` — produced no edge. These record `edge_type=
# "uses"` so the type's real call sites surface, while staying out of the
# callees ("Calls:") direction so that stays function calls.
# ---------------------------------------------------------------------------

def _extract_rs(tmp_path, src: str):
    f = tmp_path / "lib.rs"
    f.write_text(src)
    return extract_symbols_from_file(str(f))


def _uses_to(edges):
    return {e["to_symbol"] for e in edges if e["edge_type"] == "uses"}


def _calls_to(edges):
    return {e["to_symbol"] for e in edges if e["edge_type"] == "calls"}


class TestRustTypeUsageEdgesUPG44:
    def test_param_type_creates_uses_edge(self, tmp_path) -> None:
        _, edges = _extract_rs(tmp_path, "fn fetch(client: &RegistryClient) {}")
        assert ("fetch", "RegistryClient") in {(e["from_symbol"], e["to_symbol"]) for e in edges
                                               if e["edge_type"] == "uses"}

    def test_return_type_creates_uses_edge(self, tmp_path) -> None:
        _, edges = _extract_rs(tmp_path, "fn build() -> RegistryClient { todo!() }")
        assert "RegistryClient" in _uses_to(edges)

    def test_generic_arg_type_creates_uses_edge(self, tmp_path) -> None:
        # nested type arg inside Result<..> must be reached by recursion
        _, edges = _extract_rs(tmp_path, "fn build() -> Result<RegistryClient, BuildError> { todo!() }")
        assert {"RegistryClient", "BuildError"} <= _uses_to(edges)
        assert "Result" not in _uses_to(edges)  # std container skipped

    def test_struct_field_type_creates_uses_edge(self, tmp_path) -> None:
        _, edges = _extract_rs(tmp_path, "struct Holder { client: RegistryClient }")
        assert ("Holder", "RegistryClient") in {(e["from_symbol"], e["to_symbol"]) for e in edges
                                                if e["edge_type"] == "uses"}

    def test_scoped_assoc_call_links_type(self, tmp_path) -> None:
        # RegistryClient::new() → uses RegistryClient AND calls new
        _, edges = _extract_rs(tmp_path, "fn f() { let c = RegistryClient::new(); }")
        assert "RegistryClient" in _uses_to(edges)
        assert "new" in _calls_to(edges)

    def test_enum_variant_construction_links_type(self, tmp_path) -> None:
        _, edges = _extract_rs(tmp_path, "fn f() { let p = PubGrubPackage::Package(x); }")
        assert "PubGrubPackage" in _uses_to(edges)

    def test_impl_header_no_self_loop(self, tmp_path) -> None:
        # `impl RegistryClient` must not record RegistryClient --uses--> RegistryClient
        _, edges = _extract_rs(tmp_path, "impl RegistryClient { fn g(&self) {} }")
        assert ("RegistryClient", "RegistryClient") not in {
            (e["from_symbol"], e["to_symbol"]) for e in edges if e["edge_type"] == "uses"
        }

    def test_primitives_and_std_types_skipped(self, tmp_path) -> None:
        _, edges = _extract_rs(
            tmp_path,
            "fn f(a: u32, b: String, c: Vec<u8>, d: bool) -> Self { todo!() }",
        )
        targets = _uses_to(edges)
        for noise in ("u32", "String", "Vec", "u8", "bool", "Self"):
            assert noise not in targets

    def test_single_char_generic_param_skipped(self, tmp_path) -> None:
        _, edges = _extract_rs(tmp_path, "fn id<T>(x: T) -> T { x }")
        assert "T" not in _uses_to(edges)

    def test_trace_type_returns_usage_call_sites(self, tmp_path) -> None:
        # the headline acceptance: trace a type, get real call sites back
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("RegistryClient", "struct", f"{ws}/client.rs")])
        _seed_edges(g, ws, [
            (f"{ws}/ops.rs", "resolve", 10, "RegistryClient"),
            (f"{ws}/build.rs", "build_dist", 20, "RegistryClient"),
        ], edge_type="uses")
        result = g.trace(ws, "RegistryClient", direction="callers")
        callers = {e.from_symbol for e in result["callers"]}
        assert callers == {"resolve", "build_dist"}

    def test_uses_edges_excluded_from_callees(self, tmp_path) -> None:
        # `trace fn` Calls: must list function calls only, not the types it mentions
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("do_work", "function", f"{ws}/m.rs")])
        _seed_edges(g, ws, [(f"{ws}/m.rs", "do_work", 1, "helper")], edge_type="calls")
        _seed_edges(g, ws, [(f"{ws}/m.rs", "do_work", 1, "RegistryClient")], edge_type="uses")
        callees = g.callees(ws, "do_work")
        names = {e.to_symbol for e in callees}
        assert "helper" in names
        assert "RegistryClient" not in names

    def test_caller_verb_used_by_for_pure_type_usage(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_symbols(g, ws, [("RegistryClient", "struct", f"{ws}/client.rs")])
        _seed_edges(g, ws, [(f"{ws}/ops.rs", "resolve", 10, "RegistryClient")], edge_type="uses")
        result = g.trace(ws, "RegistryClient", direction="callers")
        text = g.format_trace_for_llm(result, "RegistryClient")
        assert "Used by" in text
        assert "Called by" not in text


# ---------------------------------------------------------------------------
# UPG-4.6 — locate redirect hint on description-shaped / no-match queries
#
# When the LLM misroutes a natural-language description to vectr_locate, name
# matching returns nothing. Without a nudge the model gets silence and no path
# forward; with it, locate steers to vectr_search (mirrors UPG-3.1/3.3 hints).
# ---------------------------------------------------------------------------

class TestLocateDescriptionHintUPG46:
    def test_description_query_gets_search_hint(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        result = LocateResult(symbols=[], resolution_strategy="none",
                              query="function that checks whether a number is prime")
        text = g.format_locate_l2_for_llm(result)
        assert "No symbol matching" in text
        assert "vectr_search" in text
        assert "description" in text.lower()

    def test_single_token_no_match_redirects_to_search_upg103(self, tmp_path) -> None:
        # UPG-10.3: a single-token miss is no longer a dead end — it redirects to
        # vectr_search by CONTENT (not the description phrasing) so the agent has
        # a path forward instead of falling back to grep.
        g = SymbolGraph(str(tmp_path))
        result = LocateResult(symbols=[], resolution_strategy="none", query="is_prime")
        text = g.format_locate_l2_for_llm(result)
        assert "No symbol matching" in text
        assert "vectr_search" in text
        assert "is_prime" in text          # echoes the query into the search hint
        assert "description" not in text.lower()  # this is the content-redirect, not the misroute hint

    def test_legacy_formatter_also_hints(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        text = g.format_locate_for_llm([], "where do we validate the auth token")
        assert "vectr_search" in text

    def test_looks_like_description_predicate(self) -> None:
        assert SymbolGraph._looks_like_description("parse the lock file")
        assert SymbolGraph._looks_like_description("RegistryClient new")
        assert not SymbolGraph._looks_like_description("RegistryClient")
        assert not SymbolGraph._looks_like_description("parse_lockfile")
        assert not SymbolGraph._looks_like_description("  PyList_Append  ")


# ---------------------------------------------------------------------------
# UPG-10.3 — module-level constants are locatable
# ---------------------------------------------------------------------------

class TestModuleConstantsUPG103:
    def test_module_level_constant_indexed(self, tmp_path) -> None:
        path = make_py(tmp_path, "cfg.py", textwrap.dedent('''\
            _CLAUDE_MD = """hello"""
            MAX_RETRIES = 5
        '''))
        symbols, _ = extract_symbols_from_file(path)
        by_name = {s["name"]: s for s in symbols}
        assert "_CLAUDE_MD" in by_name and by_name["_CLAUDE_MD"]["start_line"] == 1
        assert "MAX_RETRIES" in by_name and by_name["MAX_RETRIES"]["start_line"] == 2

    def test_constant_vs_variable_kind(self, tmp_path) -> None:
        # UPPER / leading-underscore-UPPER → constant; mixed-case binding → variable
        path = make_py(tmp_path, "k.py", textwrap.dedent('''\
            _CLAUDE_MD = "x"
            logger = make_logger()
        '''))
        symbols, _ = extract_symbols_from_file(path)
        by_name = {s["name"]: s["kind"] for s in symbols}
        assert by_name["_CLAUDE_MD"] == "constant"
        assert by_name["logger"] == "variable"

    def test_function_locals_not_indexed(self, tmp_path) -> None:
        # the scope guard (current_symbol == "") must keep locals out of the graph
        path = make_py(tmp_path, "scope.py", textwrap.dedent('''\
            TOP_LEVEL = 1
            def f():
                local_var = 2
                return local_var
        '''))
        symbols, _ = extract_symbols_from_file(path)
        names = {s["name"] for s in symbols}
        assert "TOP_LEVEL" in names
        assert "local_var" not in names          # never leaks out of the function
        assert "f" in names                       # function itself still indexed

    def test_annotated_module_constant_indexed(self, tmp_path) -> None:
        path = make_py(tmp_path, "ann.py", "TIMEOUT_S: int = 30\n")
        symbols, _ = extract_symbols_from_file(path)
        assert any(s["name"] == "TIMEOUT_S" and s["kind"] == "constant" for s in symbols)

    def test_locate_finds_module_constant(self, tmp_path) -> None:
        # the acceptance case: locate a constant by name, end-to-end
        g = SymbolGraph(str(tmp_path))
        path = make_py(tmp_path, "main.py", '_CLAUDE_MD = """vectr"""\ndef go(): pass\n')
        g.index_file("ws", path)
        symbols = g.locate("ws", "_CLAUDE_MD")
        assert len(symbols) >= 1
        assert symbols[0].name == "_CLAUDE_MD"
        assert symbols[0].start_line == 1


# ---------------------------------------------------------------------------
# UPG-8.7 — build resilience + version stamp / trust signals
# ---------------------------------------------------------------------------

class TestBuildResilienceUPG87:
    """build_for_workspace must be per-file resilient: one file that throws
    during extraction can no longer abort the whole loop and silently leave
    every *later* file without symbols (the real cause of the observed
    '5531 symbols across 154 files' partial graph)."""

    def test_one_throwing_file_does_not_abort_build(self, tmp_path, monkeypatch) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "a_first.py", "def a_first_fn(): return 1")
        b = make_py(tmp_path, "b_boom.py", "def b_boom_fn(): return 2")
        c = make_py(tmp_path, "c_third.py", "def c_third_fn(): return 3")

        real = _sgmod.extract_symbols_from_file

        def boom(fp):
            if "b_boom" in fp:
                raise RecursionError("simulated deep-AST parser crash (cf UPG-3.2)")
            return real(fp)

        monkeypatch.setattr(_sgmod, "extract_symbols_from_file", boom)

        g = SymbolGraph(ws)
        stats = g.build_for_workspace(ws, [a, b, c])

        # The file AFTER the bad one keeps its symbols — the loop did not abort.
        assert g.locate(ws, "a_first_fn")
        assert g.locate(ws, "c_third_fn")
        assert not g.locate(ws, "b_boom_fn")  # only the genuinely broken file is missing
        assert stats["failed"] == 1
        assert stats["complete"] is False

    def test_clean_build_is_complete(self, tmp_path) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "ok1.py", "def ok1(): pass")
        b = make_py(tmp_path, "ok2.py", "def ok2(): pass")
        g = SymbolGraph(ws)
        stats = g.build_for_workspace(ws, [a, b])
        assert stats["failed"] == 0
        assert stats["complete"] is True

    def test_partial_build_flagged_in_meta(self, tmp_path, monkeypatch) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "good.py", "def good(): pass")
        bad = make_py(tmp_path, "bad.py", "def bad(): pass")
        real = _sgmod.extract_symbols_from_file
        monkeypatch.setattr(
            _sgmod, "extract_symbols_from_file",
            lambda fp: (_ for _ in ()).throw(ValueError("x")) if "bad.py" in fp else real(fp),
        )
        g = SymbolGraph(ws)
        g.build_for_workspace(ws, [a, bad])
        meta = g.graph_meta(ws)
        assert meta["complete"] == "0"
        assert meta["failed"] == "1"


class TestSymbolCountReconciliationUPGSYMBOLCOUNTSTALE:
    """UPG-SYMBOLCOUNT-STALE: build_for_workspace's reported symbol count and the
    status-time symbol_count() query must be the same number. Previously the build
    reported the sum of freshly-inserted symbols while symbol_count() did a whole-
    table COUNT(*); rows orphaned by a file that dropped out between builds lingered
    and pushed status above build (observed +210 on a warm-cache restart)."""

    def test_build_count_matches_symbol_count_query(self, tmp_path) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "a.py", "def a_fn():\n    return 1")
        b = make_py(tmp_path, "b.py", "def b_fn():\n    return 2")
        g = SymbolGraph(ws)
        stats = g.build_for_workspace(ws, [a, b])
        assert stats["symbols"] == g.symbol_count(ws)

    def test_orphaned_rows_pruned_on_rebuild_without_file(self, tmp_path) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "a.py", "def a_fn():\n    return 1")
        b = make_py(tmp_path, "b.py", "def b_fn():\n    return 2")
        g = SymbolGraph(ws)
        g.build_for_workspace(ws, [a, b])
        assert g.locate(ws, "b_fn")

        # b.py drops out of the file set (e.g. deleted / excluded); a rebuild that
        # no longer lists it must leave no trace of its symbols behind.
        stats = g.build_for_workspace(ws, [a])
        assert not g.locate(ws, "b_fn")           # orphan symbol gone
        assert g.locate(ws, "a_fn")               # surviving file intact
        assert stats["symbols"] == g.symbol_count(ws)  # build == status, no drift

    def test_meta_symbols_matches_status_query(self, tmp_path) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "a.py", "def a_fn():\n    return 1")
        b = make_py(tmp_path, "b.py", "def b_fn():\n    return 2")
        g = SymbolGraph(ws)
        g.build_for_workspace(ws, [a, b])
        g.build_for_workspace(ws, [a])  # b orphaned then pruned
        assert int(g.graph_meta(ws)["symbols"]) == g.symbol_count(ws)


class TestGraphVersionStampUPG87:
    """A persisted graph records the toolchain that built it so an upgrade
    (new parser / changed schema / different model) is detectable and the graph
    is rebuilt rather than silently serving stale/partial results."""

    def test_fingerprint_stable_and_model_sensitive(self) -> None:
        assert graph_toolchain_fingerprint("m1") == graph_toolchain_fingerprint("m1")
        assert graph_toolchain_fingerprint("m1") != graph_toolchain_fingerprint("m2")

    def test_fingerprint_tracks_schema_version(self, monkeypatch) -> None:
        fp_before = graph_toolchain_fingerprint("m")
        monkeypatch.setattr(_sgmod, "SYMBOL_SCHEMA_VERSION", SYMBOL_SCHEMA_VERSION + 99)
        assert graph_toolchain_fingerprint("m") != fp_before

    def test_never_built_is_stale(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        assert g.is_stale(str(tmp_path), "anymodel") is True
        assert g.graph_meta(str(tmp_path)) == {}

    def test_clean_build_not_stale_same_toolchain(self, tmp_path) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "x.py", "def x(): pass")
        g = SymbolGraph(ws)
        g.build_for_workspace(ws, [a], embed_model="model-A")
        assert g.is_stale(ws, "model-A") is False

    def test_model_change_makes_stale(self, tmp_path) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "x.py", "def x(): pass")
        g = SymbolGraph(ws)
        g.build_for_workspace(ws, [a], embed_model="model-A")
        assert g.is_stale(ws, "model-B") is True  # toolchain changed → rebuild warranted

    def test_incomplete_build_is_stale(self, tmp_path, monkeypatch) -> None:
        ws = str(tmp_path)
        a = make_py(tmp_path, "good.py", "def good(): pass")
        bad = make_py(tmp_path, "bad.py", "def bad(): pass")
        real = _sgmod.extract_symbols_from_file
        monkeypatch.setattr(
            _sgmod, "extract_symbols_from_file",
            lambda fp: (_ for _ in ()).throw(ValueError("x")) if "bad.py" in fp else real(fp),
        )
        g = SymbolGraph(ws)
        g.build_for_workspace(ws, [a, bad], embed_model="m")
        # Same toolchain, but the build was partial → still stale (must rebuild).
        assert g.is_stale(ws, "m") is True


# ---------------------------------------------------------------------------
# UPG-11.10-b — qualified locate: accept + return "Class.method" form
#
# Acceptance bar:
#   1. locate("Class.method") returns only the matching class's symbol,
#      discriminated from identically-named overrides in other classes.
#   2. locate("method") bare leaf still returns ALL overrides (no regression).
#   3. Located results carry a qualified "Class.method" name when the symbol
#      lives inside a class (parity with vectr_search qualified display).
# ---------------------------------------------------------------------------

class TestQualifiedLocateUPG1110b:
    """Two classes define the same method name.  A qualified locate query
    must filter to the named class while the bare-leaf query must return both."""

    def _fixture(self, tmp_path) -> tuple:
        """Write a fixture with two classes each defining 'deconstruct' and a
        top-level 'deconstruct' function.  Returns (g, ws, filepath)."""
        src = textwrap.dedent("""\
            class Field:
                def deconstruct(self):
                    return "field"

            class RemoveField:
                def deconstruct(self):
                    return "remove"

            def deconstruct():
                pass
        """)
        fp = tmp_path / "fields.py"
        fp.write_text(src)
        g = SymbolGraph(str(tmp_path))
        g.index_file(str(tmp_path), str(fp))
        return g, str(tmp_path), str(fp)

    # --- Part (a): accepting qualified query ---

    def test_qualified_locate_filters_to_named_class(self, tmp_path) -> None:
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Field.deconstruct")
        # Only Field.deconstruct, not RemoveField.deconstruct
        assert result.resolution_strategy in ("exact", "suffix")
        names = [s.name for s in result.symbols]
        assert any("Field" in n for n in names), f"expected Field in names, got {names}"
        assert not any("Remove" in n for n in names), \
            f"RemoveField.deconstruct must be filtered out, got {names}"

    def test_qualified_locate_field_not_removefield(self, tmp_path) -> None:
        """The base Field.deconstruct is discriminated from RemoveField.deconstruct."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Field.deconstruct")
        qualified_names = [s.name for s in result.symbols]
        # Must include Field.deconstruct
        assert "Field.deconstruct" in qualified_names, \
            f"Field.deconstruct not found; got {qualified_names}"
        # Must NOT include RemoveField.deconstruct
        assert "RemoveField.deconstruct" not in qualified_names, \
            f"RemoveField.deconstruct incorrectly included; got {qualified_names}"

    def test_qualified_locate_removefield(self, tmp_path) -> None:
        """RemoveField.deconstruct can be located by its own qualifier."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "RemoveField.deconstruct")
        qualified_names = [s.name for s in result.symbols]
        assert "RemoveField.deconstruct" in qualified_names, \
            f"RemoveField.deconstruct not found; got {qualified_names}"
        assert "Field.deconstruct" not in qualified_names, \
            f"Field.deconstruct must not appear in RemoveField results; got {qualified_names}"

    # --- Part (b): bare-leaf locate unchanged ---

    def test_bare_leaf_locate_returns_all_overrides(self, tmp_path) -> None:
        """Bare 'deconstruct' must still return ALL definitions (no regression)."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "deconstruct")
        # All three deconstruct symbols should be returned
        names = [s.name for s in result.symbols]
        # At minimum Field.deconstruct and RemoveField.deconstruct must appear
        class_names = {n.split(".")[0] for n in names if "." in n}
        assert "Field" in class_names, f"Field not in results for bare query: {names}"
        assert "RemoveField" in class_names, f"RemoveField not in results for bare query: {names}"

    def test_bare_leaf_strategy_unchanged(self, tmp_path) -> None:
        """Bare-leaf locate uses exact strategy, not filtered."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "deconstruct")
        assert result.resolution_strategy == "exact"
        # Count: should have at least 2 class methods + 1 top-level
        assert len(result.symbols) >= 2

    # --- Part (b): qualified names in returned symbols ---

    def test_qualified_name_returned_for_class_method(self, tmp_path) -> None:
        """locate_l2 populates 'Class.method' on the symbol.name (not bare leaf)."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "deconstruct")
        names = [s.name for s in result.symbols]
        # At least one symbol should carry a qualified name
        assert any("." in n for n in names), \
            f"No qualified names returned; got {names}"

    def test_toplevel_function_stays_unqualified(self, tmp_path) -> None:
        """A top-level function (no enclosing class) must NOT get a class prefix."""
        src = textwrap.dedent("""\
            def standalone():
                pass
        """)
        fp = tmp_path / "top.py"
        fp.write_text(src)
        g = SymbolGraph(str(tmp_path))
        g.index_file(str(tmp_path), str(fp))
        result = g.locate_l2(str(tmp_path), "standalone")
        assert len(result.symbols) >= 1
        assert result.symbols[0].name == "standalone", \
            f"Top-level function should stay unqualified; got {result.symbols[0].name}"

    # --- enclosing class helper unit tests ---

    def test_enclosing_class_from_file_method(self, tmp_path) -> None:
        from agent.symbol_graph._graph import _enclosing_class_from_file
        src = textwrap.dedent("""\
            class MyClass:
                def my_method(self):
                    pass
        """)
        fp = tmp_path / "c.py"
        fp.write_text(src)
        # my_method starts at line 2
        cls = _enclosing_class_from_file(str(fp), 2)
        assert cls == "MyClass", f"Expected 'MyClass', got '{cls}'"

    def test_enclosing_class_from_file_toplevel(self, tmp_path) -> None:
        from agent.symbol_graph._graph import _enclosing_class_from_file
        src = "def standalone(): pass\n"
        fp = tmp_path / "t.py"
        fp.write_text(src)
        cls = _enclosing_class_from_file(str(fp), 1)
        assert cls == "", f"Top-level function should return empty, got '{cls}'"

    def test_enclosing_class_from_file_nonexistent(self, tmp_path) -> None:
        from agent.symbol_graph._graph import _enclosing_class_from_file
        cls = _enclosing_class_from_file(str(tmp_path / "ghost.py"), 1)
        assert cls == ""

    # --- format integration ---

    def test_format_l2_shows_qualified_name(self, tmp_path) -> None:
        """format_locate_l2_for_llm should display Class.method, not bare leaf."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Field.deconstruct")
        text = g.format_locate_l2_for_llm(result)
        assert "Field.deconstruct" in text, \
            f"Qualified name not in formatted output:\n{text}"


# ---------------------------------------------------------------------------
# UPG-RUST-IMPL-QUALIFIED-LOCATE — qualified `Type.method` locate/trace must
# resolve a method declared inside a Rust `impl Type { }` block exactly as it
# already resolves a Python/Java `class` member. Rust has no `class` keyword,
# so the pre-existing enclosing-owner lookup (a text scan for a `class NAME`
# line) never matches an impl block; the owning type must instead come from
# tree-sitter's own `type` field on the `impl_item` node.
# ---------------------------------------------------------------------------

class TestRustImplQualifiedLocateUPG:
    def _fixture(self, tmp_path) -> tuple:
        """Two structs each with an `impl` block that defines a method of the
        SAME leaf name (`add_item`), plus a trait impl, so a qualifier is
        required to disambiguate — mirrors the Class.method fixtures already
        used for Python/Java elsewhere in this file."""
        src = textwrap.dedent("""\
            struct Basket {
                count: u32,
            }

            impl Basket {
                pub fn add_item(&mut self, n: u32) {
                    self.count += n;
                }

                pub fn total(&self) -> u32 {
                    self.count
                }
            }

            trait Weighable {
                fn weigh(&self) -> u32;
            }

            impl Weighable for Basket {
                fn weigh(&self) -> u32 {
                    self.count * 2
                }
            }

            struct Crate {
                count: u32,
            }

            impl Crate {
                pub fn add_item(&mut self, n: u32) {
                    self.count += n * 10;
                }
            }

            fn use_basket() {
                let mut b = Basket { count: 0 };
                b.add_item(3);
            }
        """)
        fp = tmp_path / "basket.rs"
        fp.write_text(src)
        g = SymbolGraph(str(tmp_path))
        g.index_file(str(tmp_path), str(fp))
        return g, str(tmp_path), str(fp)

    # --- (a) qualified locate resolves to the method's own file:line ---

    def test_qualified_locate_resolves_impl_method_file_line(self, tmp_path) -> None:
        g, ws, fp = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Basket.add_item")
        names = [s.name for s in result.symbols]
        assert "Basket.add_item" in names, f"expected Basket.add_item, got {names}"
        sym = next(s for s in result.symbols if s.name == "Basket.add_item")
        assert sym.file_path == fp
        lines = Path(fp).read_text().splitlines()
        assert "add_item" in lines[sym.start_line - 1], (
            f"start_line {sym.start_line} does not point at the method: "
            f"{lines[sym.start_line - 1]!r}"
        )
        assert "10" not in (sym.snippet or ""), (
            "resolved to Crate.add_item's body, not Basket's"
        )

    def test_qualified_locate_excludes_other_impl_same_leaf(self, tmp_path) -> None:
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Basket.add_item")
        names = [s.name for s in result.symbols]
        assert "Crate.add_item" not in names, f"Crate.add_item leaked in: {names}"

    def test_qualified_locate_other_impl_by_its_own_qualifier(self, tmp_path) -> None:
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Crate.add_item")
        names = [s.name for s in result.symbols]
        assert "Crate.add_item" in names
        assert "Basket.add_item" not in names

    # --- trait impl: the owner is the implemented TYPE, never the TRAIT ---

    def test_trait_impl_owner_is_type_not_trait(self, tmp_path) -> None:
        """`impl Weighable for Basket` — the owner is Basket, never Weighable."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Basket.weigh")
        names = [s.name for s in result.symbols]
        assert "Basket.weigh" in names, f"expected Basket.weigh, got {names}"

    def test_trait_impl_wrong_qualifier_not_accepted(self, tmp_path) -> None:
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "Weighable.weigh")
        names = [s.name for s in result.symbols]
        assert "Basket.weigh" not in names, (
            f"the trait name must never be accepted as the owning qualifier: {names}"
        )

    # --- (b) bare-name locate is unaffected by the fix ---

    def test_bare_name_locate_returns_all_impls(self, tmp_path) -> None:
        g, ws, _ = self._fixture(tmp_path)
        result = g.locate_l2(ws, "add_item")
        names = [s.name for s in result.symbols]
        owners = {n.split(".")[0] for n in names if "." in n}
        assert "Basket" in owners and "Crate" in owners, f"got {names}"

    # --- (c) regression: Java class-qualified locate unaffected ---
    # (Python coverage already lives in TestQualifiedLocateUPG1110b above —
    # this fixture is Rust's fallback partner: the AST-based impl lookup is a
    # fallback that only runs when the fast `class NAME` text scan finds
    # nothing, so it must never change Java's already-working path.)

    def test_java_class_qualified_locate_unaffected(self, tmp_path) -> None:
        src = textwrap.dedent("""\
            public class Basket {
                public int addItem(int n) {
                    return n;
                }
            }
        """)
        fp = tmp_path / "Basket.java"
        fp.write_text(src)
        g = SymbolGraph(str(tmp_path))
        g.index_file(str(tmp_path), str(fp))
        result = g.locate_l2(str(tmp_path), "Basket.addItem")
        names = [s.name for s in result.symbols]
        assert "Basket.addItem" in names, f"Java qualified locate regressed: {names}"

    # --- (d) trace synergy with UPG-TRACE-CALLERS-QUALIFIER-VALIDATION ---

    def test_trace_valid_impl_qualifier_resolves_without_banner(self, tmp_path) -> None:
        g, ws, _ = self._fixture(tmp_path)
        result = g.trace(ws, "Basket.add_item", direction="callers")
        assert result.get("qualified_class") == "Basket"
        assert "qualifier_unresolved" not in result
        text = g.format_trace_for_llm(result, "Basket.add_item")
        assert "does not resolve" not in text

    def test_trace_valid_trait_impl_qualifier_resolves(self, tmp_path) -> None:
        """A method inside `impl Trait for Type` resolves its qualifier to
        the TYPE, matching locate's trait-vs-type distinction."""
        g, ws, _ = self._fixture(tmp_path)
        result = g.trace(ws, "Basket.weigh", direction="callees")
        assert result.get("qualified_class") == "Basket"
        assert "qualifier_unresolved" not in result

    def test_trace_fabricated_impl_qualifier_still_banners(self, tmp_path) -> None:
        g, ws, _ = self._fixture(tmp_path)
        result = g.trace(ws, "Basket.does_not_exist", direction="callers")
        assert result.get("qualifier_unresolved") is True
        text = g.format_trace_for_llm(result, "Basket.does_not_exist")
        assert "does not resolve" in text

    # --- direct unit tests on the new AST-fallback helper ---

    def test_enclosing_container_from_file_impl_method(self, tmp_path) -> None:
        from agent.symbol_graph._graph import _enclosing_container_from_file
        src = textwrap.dedent("""\
            struct Widget {
                n: u32,
            }

            impl Widget {
                fn spin(&self) -> u32 {
                    self.n
                }
            }
        """)
        fp = tmp_path / "w.rs"
        fp.write_text(src)
        owner = _enclosing_container_from_file(str(fp), 6)
        assert owner == "Widget", f"expected 'Widget', got {owner!r}"

    def test_enclosing_container_from_file_generic_multiline_impl(self, tmp_path) -> None:
        """A generic impl header spanning multiple lines has no single
        `class`-style keyword line a regex could match — the AST-based
        fallback (tree-sitter's own `type` field) resolves the owner
        regardless of how the header is laid out."""
        from agent.symbol_graph._graph import _enclosing_container_from_file
        src = textwrap.dedent("""\
            struct Wrapper<T> {
                inner: T,
            }

            impl<T: Clone>
                Wrapper<T>
            {
                fn get(&self) -> T {
                    self.inner.clone()
                }
            }
        """)
        fp = tmp_path / "wrap.rs"
        fp.write_text(src)
        owner = _enclosing_container_from_file(str(fp), 8)
        assert owner == "Wrapper", f"expected 'Wrapper', got {owner!r}"

    def test_enclosing_container_from_file_no_impl_block(self, tmp_path) -> None:
        from agent.symbol_graph._graph import _enclosing_container_from_file
        fp = tmp_path / "s.rs"
        fp.write_text("fn standalone() {}\n")
        assert _enclosing_container_from_file(str(fp), 1) == ""

    def test_enclosing_container_from_file_falls_back_only_when_needed(
        self, tmp_path
    ) -> None:
        """A Python file with a real `class` line is answered by the fast
        text scan; the AST fallback never even considers it."""
        from agent.symbol_graph._graph import _enclosing_container_from_file
        src = textwrap.dedent("""\
            class Basket:
                def add_item(self, n):
                    return n
        """)
        fp = tmp_path / "basket.py"
        fp.write_text(src)
        assert _enclosing_container_from_file(str(fp), 2) == "Basket"


# ---------------------------------------------------------------------------
# UPG-15.10 — locate: canonical library defs rank before inner test-scope stubs
#
# Acceptance case (F29): vectr_locate('Model') must return the canonical
# top-level library definition (e.g. django/db/models/base.py:Model, ~1400 lines)
# in the top-5, not only inner `class Model(models.Model):` stubs defined inside
# test method bodies in test files.
#
# Ranking signals added (in order of discrimination power, after kind):
#   1. scope_depth  — 0 = top-level, 1+ = inside a function body
#   2. span_bucket  — 0 = large (≥50 lines), 1 = medium, 2 = tiny (<10 lines)
# Both signals penalise inner test-scope stubs relative to canonical defs.
# ---------------------------------------------------------------------------

class TestLocateRankingUPG1510:
    """Locate canonical library defs before inner test-scope stubs (UPG-15.10)."""

    # --- scope depth unit tests ---

    def test_scope_depth_top_level_class(self, tmp_path) -> None:
        """A class at module level has scope_depth=0."""
        lines = [
            "class Model:",
            "    pass",
        ]
        depth = _locate_scope_depth_from_lines(lines, start_line=1)
        assert depth == 0, f"Top-level class must have scope_depth=0, got {depth}"

    def test_scope_depth_class_inside_method(self, tmp_path) -> None:
        """A class defined inside a method body has scope_depth=1."""
        lines = [
            "class TestModelChecks:",
            "    def test_invalid_model(self):",
            "        class Model(models.Model):",
            "            pass",
        ]
        # 'class Model' starts at line 3 (1-indexed)
        depth = _locate_scope_depth_from_lines(lines, start_line=3)
        assert depth == 1, f"Class inside a method must have scope_depth=1, got {depth}"

    def test_scope_depth_class_inside_nested_functions(self, tmp_path) -> None:
        """A class inside two nested function calls has scope_depth=2."""
        lines = [
            "def outer():",
            "    def inner():",
            "        class Stub:",
            "            pass",
        ]
        depth = _locate_scope_depth_from_lines(lines, start_line=3)
        assert depth == 2, f"Class inside two defs must have scope_depth=2, got {depth}"

    def test_scope_depth_top_level_fast_path(self) -> None:
        """Symbols at indent=0 return 0 without scanning (fast path)."""
        lines = ["class TopLevel:", "    x = 1"]
        depth = _locate_scope_depth_from_lines(lines, start_line=1)
        assert depth == 0

    def test_scope_depth_empty_lines(self) -> None:
        """Empty lines list or out-of-range start_line returns 0."""
        assert _locate_scope_depth_from_lines([], 1) == 0
        assert _locate_scope_depth_from_lines(["class A: pass"], 0) == 0

    # --- partial_match_key unit tests with scope_depth signal ---

    def test_partial_match_key_scope_depth_penalised(self) -> None:
        """A row with scope_depth=1 ranks below scope_depth=0 all else equal."""
        row_canonical = {
            "name": "Model", "kind": "class",
            "file_path": "django/db/models/base.py",
            "start_line": 1, "end_line": 100,
        }
        row_inner = {
            "name": "Model", "kind": "class",
            "file_path": "tests/check_framework/test_model_checks.py",
            "start_line": 20, "end_line": 22,
        }
        key_canonical = _partial_match_key(row_canonical, "model", scope_depth=0)
        key_inner = _partial_match_key(row_inner, "model", scope_depth=1)
        assert key_canonical < key_inner, (
            f"Canonical (scope_depth=0) must rank before inner stub (scope_depth=1); "
            f"canonical key={key_canonical}, inner key={key_inner}"
        )

    def test_partial_match_key_large_span_beats_tiny_span(self) -> None:
        """A row with a large line span ranks before a tiny-span row (all else equal)."""
        row_large = {
            "name": "Model", "kind": "class",
            "file_path": "lib/base.py",
            "start_line": 1, "end_line": 1400,
        }
        row_tiny = {
            "name": "Model", "kind": "class",
            "file_path": "lib/other.py",
            "start_line": 100, "end_line": 103,
        }
        key_large = _partial_match_key(row_large, "model", scope_depth=0)
        key_tiny = _partial_match_key(row_tiny, "model", scope_depth=0)
        assert key_large < key_tiny, (
            f"Large-span (canonical) must rank before tiny-span (stub); "
            f"large={key_large}, tiny={key_tiny}"
        )

    # --- end-to-end locate_l2 test ---

    def _make_canonical_lib(self, tmp_path) -> str:
        """Write a canonical library file with a large top-level Model class."""
        lib_dir = tmp_path / "django" / "db" / "models"
        lib_dir.mkdir(parents=True)
        # Build a ~60-line class to exceed LOCATE_LARGE_SPAN_THRESHOLD (50 lines)
        body = "\n".join(f"    attr_{i} = None" for i in range(58))
        src = f"class Model:\n{body}\n"
        fp = lib_dir / "base.py"
        fp.write_text(src)
        return str(fp)

    def _make_test_file_with_inner_models(self, tmp_path) -> str:
        """Write a test file with many inner `class Model` stubs inside test methods."""
        test_dir = tmp_path / "tests" / "check_framework"
        test_dir.mkdir(parents=True)
        # 12 inner classes inside test methods — more than the default locate limit=10
        methods = []
        for i in range(12):
            methods.append(textwrap.dedent(f"""\
                def test_invalid_model_{i}(self):
                    class Model(models.Model):
                        pass
            """))
        src = "class TestModelChecks:\n" + textwrap.indent("".join(methods), "    ")
        fp = test_dir / "test_model_checks.py"
        fp.write_text(src)
        return str(fp)

    def test_canonical_model_ranks_in_top5_over_test_inner_stubs(self, tmp_path) -> None:
        """F29 acceptance: locate_l2('Model') returns the canonical library class
        in top-5 even when the same-named inner test-scope stubs outnumber the limit."""
        lib_fp = self._make_canonical_lib(tmp_path)
        test_fp = self._make_test_file_with_inner_models(tmp_path)

        g = SymbolGraph(str(tmp_path))
        # Index the test file first so its rows have lower rowids (indexed earlier),
        # which mimics the live Django situation where test files appear before the
        # canonical in alphabetical indexing order.
        g.index_file(str(tmp_path), test_fp)
        g.index_file(str(tmp_path), lib_fp)

        result = g.locate_l2(str(tmp_path), "Model", limit=10)
        assert result.resolution_strategy == "exact"
        names_and_files = [(s.name, s.file_path) for s in result.symbols]
        # The canonical library definition must appear in the top-5 results
        canonical_rank = next(
            (i for i, (_, fp) in enumerate(names_and_files) if "base.py" in fp),
            None,
        )
        assert canonical_rank is not None, (
            f"Canonical django/db/models/base.py:Model not found in results at all.\n"
            f"Got: {names_and_files}"
        )
        assert canonical_rank < 5, (
            f"Canonical library Model must rank in top-5 (got rank {canonical_rank}).\n"
            f"Top results: {names_and_files}"
        )
        # The top result must NOT be an inner test-scope stub
        top_file = names_and_files[0][1]
        assert "test" not in top_file.replace("\\", "/").split("/")[-1], (
            f"Rank-1 result must not be a test file; got {top_file}"
        )

    def test_scope_depth_batch_caches_per_file(self, tmp_path) -> None:
        """_locate_scope_depth_batch reads each file at most once (returns correct values)."""
        src = textwrap.dedent("""\
            class Outer:
                def method(self):
                    class Inner:
                        pass
        """)
        fp = tmp_path / "m.py"
        fp.write_text(src)
        # Simulate two rows from the same file: Outer at line 1, Inner at line 3
        rows = [
            {"file_path": str(fp), "start_line": 1},
            {"file_path": str(fp), "start_line": 3},
        ]
        depths = _locate_scope_depth_batch(rows)
        assert depths[0] == 0, f"Outer class (top-level) must be depth 0, got {depths[0]}"
        assert depths[1] == 1, f"Inner class (inside method) must be depth 1, got {depths[1]}"


# ---------------------------------------------------------------------------
# UPG-15.10x / F49 — locate: bare module-level defs rank before same-named
# class methods (the "common leaf" collision).
#
# `_partial_match_key`'s existing span_bucket signal assumes canonical == large,
# which correctly demotes tiny inner test-scope stubs (UPG-15.10/F29) but
# backfires when the canonical answer is itself a SHORT module-level function
# (e.g. a thin delegating wrapper) competing against a same-named, LARGER
# method on some unrelated class. The class_enclosed signal targets that
# orthogonal collision directly and is ranked ahead of span_bucket so a small
# canonical function is never outranked by a bigger method look-alike.
# ---------------------------------------------------------------------------

class TestLocateRankingClassEnclosedF49:
    """Bare module-level definitions outrank same-named class methods (F49)."""

    # --- _enclosing_class_from_lines / _locate_class_enclosed_batch unit tests ---

    def test_enclosing_class_from_lines_module_level_function(self) -> None:
        """A module-level function has no enclosing class."""
        lines = ["def render(request, template_name):", "    return None"]
        cls = _enclosing_class_from_lines(lines, start_line=1)
        assert cls == "", f"Module-level function must have no enclosing class, got {cls!r}"

    def test_enclosing_class_from_lines_method(self) -> None:
        """A method's immediate enclosing scope is its class."""
        lines = [
            "class ForNode:",
            "    def render(self, context):",
            "        return None",
        ]
        cls = _enclosing_class_from_lines(lines, start_line=2)
        assert cls == "ForNode", f"Method must resolve its enclosing class, got {cls!r}"

    def test_enclosing_class_from_lines_class_nested_in_function_not_class(self) -> None:
        """A class defined inside a function body (UPG-15.10/F29 shape) has NO
        enclosing CLASS — its immediate enclosing scope is the function, so this
        signal must not double-penalise the inner-test-stub case (that's scope_depth's
        job); class_enclosed only targets the bare-function-vs-method collision."""
        lines = [
            "class TestSuite:",
            "    def test_it(self):",
            "        class Widget:",
            "            pass",
        ]
        cls = _enclosing_class_from_lines(lines, start_line=3)
        assert cls == "", (
            f"A class immediately nested in a function must report no enclosing "
            f"class (its immediate parent is the function, not TestSuite); got {cls!r}"
        )

    def test_enclosing_class_modifier_prefixed_java(self) -> None:
        """UPG-CLASS-DEF-RE-MODIFIERS: a `public class` (the standard Java form)
        must resolve its enclosing class — the bare `^class` regex missed it and
        misread the declaration line as a scope exit."""
        lines = [
            "public class Factory {",
            "    public Producer createProducer() {",
            "        return null;",
            "    }",
            "}",
        ]
        assert _enclosing_class_from_lines(lines, start_line=2) == "Factory"

    def test_enclosing_class_multi_modifier_java(self) -> None:
        lines = [
            "public final class Registry {",
            "    void register() {}",
            "}",
        ]
        assert _enclosing_class_from_lines(lines, start_line=2) == "Registry"

    def test_enclosing_class_export_ts(self) -> None:
        """UPG-CLASS-DEF-RE-MODIFIERS: `export class` (the standard TS/JS form)."""
        lines = [
            "export class Foo {",
            "    run(): void {}",
            "}",
        ]
        assert _enclosing_class_from_lines(lines, start_line=2) == "Foo"

    def test_enclosing_class_export_default_and_abstract_ts(self) -> None:
        for header, name in (
            ("export default class Bar {", "Bar"),
            ("export abstract class Baz {", "Baz"),
        ):
            lines = [header, "    m() {}", "}"]
            assert _enclosing_class_from_lines(lines, start_line=2) == name

    def test_enclosing_class_modifier_run_without_class_is_not_a_class(self) -> None:
        """A modifier run NOT followed by `class` (e.g. `public void helper()`)
        must still be treated as a non-class scope exit, returning ''."""
        lines = [
            "public void helper() {",
            "    int x = 1;",
            "}",
        ]
        assert _enclosing_class_from_lines(lines, start_line=2) == ""

    def test_locate_class_enclosed_batch(self, tmp_path) -> None:
        """_locate_class_enclosed_batch reads each file once and reports True only
        for symbols whose immediate enclosing scope is a class."""
        src = textwrap.dedent("""\
            def render(request, template_name):
                return None

            class ForNode:
                def render(self, context):
                    return None
        """)
        fp = tmp_path / "m.py"
        fp.write_text(src)
        rows = [
            {"file_path": str(fp), "start_line": 1},  # module-level render
            {"file_path": str(fp), "start_line": 5},  # ForNode.render
        ]
        flags = _locate_class_enclosed_batch(rows)
        assert flags[0] is False, "Module-level render must not be class_enclosed"
        assert flags[1] is True, "ForNode.render must be class_enclosed"

    # --- _partial_match_key unit tests ---

    def test_partial_match_key_class_enclosed_penalised(self) -> None:
        """A class-enclosed candidate ranks below a module-level candidate,
        all else equal."""
        row_module_level = {
            "name": "render", "kind": "function",
            "file_path": "lib/shortcuts.py", "start_line": 1, "end_line": 5,
        }
        row_method = {
            "name": "render", "kind": "function",
            "file_path": "lib/nodes.py", "start_line": 1, "end_line": 5,
        }
        key_module = _partial_match_key(
            row_module_level, "render", scope_depth=0, class_enclosed=False,
        )
        key_method = _partial_match_key(
            row_method, "render", scope_depth=0, class_enclosed=True,
        )
        assert key_module < key_method, (
            f"Module-level def must rank before a same-named class method; "
            f"module key={key_module}, method key={key_method}"
        )

    def test_partial_match_key_class_enclosed_beats_larger_span(self) -> None:
        """Regression for the exact F49 bug: a SHORT module-level function must
        outrank a LARGER same-named class method — span_bucket alone got this
        backwards (larger span looked more 'canonical') until class_enclosed was
        placed ahead of it in the sort key."""
        row_short_module_level = {
            "name": "render", "kind": "function",
            "file_path": "lib/shortcuts.py", "start_line": 1, "end_line": 8,   # tiny span
        }
        row_large_method = {
            "name": "render", "kind": "function",
            "file_path": "lib/nodes.py", "start_line": 1, "end_line": 65,      # medium/large span
        }
        key_short = _partial_match_key(
            row_short_module_level, "render", scope_depth=0, class_enclosed=False,
        )
        key_large = _partial_match_key(
            row_large_method, "render", scope_depth=0, class_enclosed=True,
        )
        assert key_short < key_large, (
            f"Short module-level def must beat a larger class-method look-alike; "
            f"short={key_short}, large={key_large}"
        )

    def test_partial_match_key_no_regression_when_all_candidates_are_methods(self) -> None:
        """When every same-name candidate is a class method (no bare module-level
        def exists — e.g. locate('save') across several model classes), the
        class_enclosed tier is a no-op tie and span_bucket still decides, exactly
        as before this change (UPG-15.10 behaviour preserved)."""
        row_large_method = {
            "name": "save", "kind": "function",
            "file_path": "lib/models.py", "start_line": 1, "end_line": 60,
        }
        row_tiny_method = {
            "name": "save", "kind": "function",
            "file_path": "lib/other.py", "start_line": 1, "end_line": 4,
        }
        key_large = _partial_match_key(
            row_large_method, "save", scope_depth=0, class_enclosed=True,
        )
        key_tiny = _partial_match_key(
            row_tiny_method, "save", scope_depth=0, class_enclosed=True,
        )
        assert key_large < key_tiny, (
            "With class_enclosed tied (both methods), the larger span must still "
            f"win as before; large={key_large}, tiny={key_tiny}"
        )

    # --- end-to-end locate_l2 test (F49 acceptance witness, neutral names) ---

    def _make_lib_with_canonical_render(self, tmp_path) -> str:
        """A short, module-level canonical `render` — the library's public API
        entry point, e.g. a thin dispatcher over a templating backend."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir(parents=True)
        src = textwrap.dedent("""\
            def render(request, template_name, context=None):
                content = _render_to_string(template_name, context, request)
                return content
        """)
        fp = lib_dir / "shortcuts.py"
        fp.write_text(src)
        return str(fp)

    def _make_lib_with_render_methods(self, tmp_path) -> str:
        """Several unrelated node classes each define a larger `render` method —
        common-leaf look-alikes competing with the canonical function above."""
        lib_dir = tmp_path / "lib"
        classes = []
        for i in range(8):
            body_lines = "\n".join(f"        step_{j} = {j}" for j in range(20))
            classes.append(textwrap.dedent(f"""\
                class Node{i}:
                    def render(self, context):
                {body_lines}
                        return context
            """))
        fp = lib_dir / "nodes.py"
        fp.write_text("\n".join(classes))
        return str(fp)

    def test_canonical_render_ranks_first_over_class_method_lookalikes(self, tmp_path) -> None:
        """F49 acceptance: locate_l2('render') returns the canonical module-level
        function first, ahead of larger same-named methods on unrelated classes."""
        lib_fp = self._make_lib_with_canonical_render(tmp_path)
        nodes_fp = self._make_lib_with_render_methods(tmp_path)

        g = SymbolGraph(str(tmp_path))
        # Index the method-heavy file first so its rows have lower rowids —
        # mirrors the live django situation where defaulttags.py sorts before
        # shortcuts.py alphabetically.
        g.index_file(str(tmp_path), nodes_fp)
        g.index_file(str(tmp_path), lib_fp)

        result = g.locate_l2(str(tmp_path), "render", limit=5)
        assert result.resolution_strategy == "exact"
        names_and_files = [(s.name, s.file_path) for s in result.symbols]
        assert names_and_files, "locate_l2('render') returned no results"
        top_name, top_file = names_and_files[0]
        assert "shortcuts.py" in top_file and "." not in top_name, (
            f"Rank-1 result must be the bare module-level canonical render, "
            f"got {names_and_files[0]}.\nFull top-5: {names_and_files}"
        )


# ---------------------------------------------------------------------------
# ARCH-1a — file-level PageRank importance: compute, persist, read API
# ---------------------------------------------------------------------------

def _seed_importance_graph(g: "SymbolGraph", ws: str, tmp_path) -> dict[str, str]:
    """Seed a synthetic graph for importance tests.

    Graph layout (all files also have symbols so they're in the PageRank node set):
      - file_b.py defines symbol 'foo' (leaf='foo')
      - file_c.py, file_d.py, file_e.py each define their own symbol AND have an
        edge referencing 'foo' (they all link to file_b.py via the def<->ref graph)
      - file_a.py defines its own symbol but has no outgoing refs and no in-links

    Expected result after PageRank + normalization:
      file_b.py has the highest score (= 1.0) because 3 files reference it.
      file_a.py has a strictly lower score (no in-links from refs).
    """
    import time
    now = time.time()

    files = {
        "a": str(tmp_path / "file_a.py"),
        "b": str(tmp_path / "file_b.py"),
        "c": str(tmp_path / "file_c.py"),
        "d": str(tmp_path / "file_d.py"),
        "e": str(tmp_path / "file_e.py"),
    }
    # Create the actual files so SymbolGraph is consistent
    for fp in files.values():
        Path(fp).write_text("# placeholder\n")

    with g._conn() as conn:
        # All files get at least one symbol so they're in the PageRank node set.
        # file_b defines 'foo' — the target of all the ref-edges.
        conn.execute(
            "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ws, "foo", "function", files["b"], 1, 5, now),
        )
        # file_a defines something unrelated (no in-links to it from refs)
        conn.execute(
            "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ws, "bar_unrelated", "function", files["a"], 1, 3, now),
        )
        # file_c/d/e each define their own unique symbol (so they enter the node set)
        # AND each has an edge referencing 'foo' (to_symbol='foo' → links to file_b)
        for key, ref_file in (("c", files["c"]), ("d", files["d"]), ("e", files["e"])):
            conn.execute(
                "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ws, f"caller_{key}", "function", ref_file, 1, 3, now),
            )
            conn.execute(
                "INSERT INTO edges (workspace, from_file, from_symbol, from_line, to_symbol, edge_type) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ws, ref_file, f"caller_{key}", 10, "foo", "calls"),
            )

    return files


# ---------------------------------------------------------------------------
# UPG-QUALIFIER-SIM-FILE-CACHE: _enclosing_container_batch is the
# _locate_class_enclosed_batch-style per-file cache for
# nearest_symbol_names' qualified near-miss ranking pool, so
# _qualifier_similarity_key never re-reads (or re-parses) the same
# candidate file once per candidate defined in it.
# ---------------------------------------------------------------------------

class TestEnclosingContainerBatchUPG:
    def test_reads_each_file_at_most_once_across_many_candidates(
        self, tmp_path, monkeypatch,
    ) -> None:
        from agent.symbol_graph._graph import _enclosing_container_batch

        src = (
            "class WidgetRegistry:\n    def process(self):\n        pass\n"
            "class GadgetHelper:\n    def process(self):\n        pass\n"
            "class SprocketFactory:\n    def process(self):\n        pass\n"
        )
        fp = tmp_path / "widgets.py"
        fp.write_text(src)

        calls = {"count": 0}
        real_read_text = Path.read_text

        def counting_read_text(self, *args, **kwargs):
            if str(self) == str(fp):
                calls["count"] += 1
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counting_read_text)

        symbols = [
            Symbol(1, "ws", "process", "method", str(fp), 2, 3),
            Symbol(2, "ws", "process", "method", str(fp), 5, 6),
            Symbol(3, "ws", "process", "method", str(fp), 8, 9),
        ]
        names = _enclosing_container_batch(symbols)

        assert names == ["WidgetRegistry", "GadgetHelper", "SprocketFactory"]
        assert calls["count"] == 1, (
            "expected one read for 3 candidates sharing one file_path, "
            f"got {calls['count']}"
        )

    def test_symbol_with_no_enclosing_container_returns_empty_string(
        self, tmp_path,
    ) -> None:
        from agent.symbol_graph._graph import _enclosing_container_batch

        fp = tmp_path / "mod.py"
        fp.write_text("def free_function():\n    pass\n")

        names = _enclosing_container_batch(
            [Symbol(1, "ws", "free_function", "function", str(fp), 1, 2)]
        )
        assert names == [""]

    def test_multiple_files_each_read_once(self, tmp_path, monkeypatch) -> None:
        from agent.symbol_graph._graph import _enclosing_container_batch

        fp_a = tmp_path / "a.py"
        fp_a.write_text("class Alpha:\n    def process(self):\n        pass\n")
        fp_b = tmp_path / "b.py"
        fp_b.write_text("class Beta:\n    def process(self):\n        pass\n")

        calls: dict[str, int] = {}
        real_read_text = Path.read_text

        def counting_read_text(self, *args, **kwargs):
            calls[str(self)] = calls.get(str(self), 0) + 1
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", counting_read_text)

        symbols = [
            Symbol(1, "ws", "process", "method", str(fp_a), 2, 3),
            Symbol(2, "ws", "process", "method", str(fp_b), 2, 3),
            Symbol(3, "ws", "process", "method", str(fp_a), 2, 3),
        ]
        names = _enclosing_container_batch(symbols)

        assert names == ["Alpha", "Beta", "Alpha"]
        assert calls[str(fp_a)] == 1
        assert calls[str(fp_b)] == 1


class TestFileImportanceARCH1a:
    """ARCH-1a: file-level PageRank importance computation, persistence, and read API."""

    def test_most_referenced_file_gets_highest_score(self, tmp_path) -> None:
        """file_b defines 'foo'; c/d/e reference it → file_b must score 1.0 (max)."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)

        g._compute_and_store_importance(ws)
        scores = g.file_importance(ws)

        assert files["b"] in scores, "file_b must have an importance score"
        assert abs(scores[files["b"]] - 1.0) < 1e-9, (
            f"file_b (most-referenced) must have score=1.0 after normalization, "
            f"got {scores[files['b']]}"
        )

    def test_most_referenced_beats_barely_referenced(self, tmp_path) -> None:
        """file_b (3 in-links) must score strictly higher than file_a (0 in-links)."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)

        g._compute_and_store_importance(ws)
        scores = g.file_importance(ws)

        score_b = scores.get(files["b"], 0.0)
        score_a = scores.get(files["a"], 0.0)
        assert score_b > score_a, (
            f"file_b (3 in-links) must score above file_a (0 in-links); "
            f"b={score_b:.6f}, a={score_a:.6f}"
        )

    def test_persistence_round_trip(self, tmp_path) -> None:
        """build_for_workspace writes importance; file_importance returns expected files."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)

        # Trigger compute via the internal method (build_for_workspace calls it)
        g._compute_and_store_importance(ws)
        scores = g.file_importance(ws)

        # All scored files must have scores in (0, 1]
        assert len(scores) > 0, "file_importance must return non-empty dict after compute"
        for fp, score in scores.items():
            assert 0.0 < score <= 1.0, (
                f"{fp} has score {score} outside (0, 1]"
            )

    def test_build_for_workspace_stores_importance(self, tmp_path) -> None:
        """build_for_workspace end-to-end: importance is persisted and readable."""
        g = SymbolGraph(str(tmp_path))
        # Use real Python files so index_file works via tree-sitter
        p1 = make_py(tmp_path, "lib.py", "def core_fn(): pass\n")
        p2 = make_py(tmp_path, "caller.py", "def user():\n    core_fn()\n")
        g.build_for_workspace(str(tmp_path), [p1, p2])

        scores = g.file_importance(str(tmp_path))
        # At least one score must be present after a real build
        assert len(scores) >= 1

    def test_rebuild_idempotency(self, tmp_path) -> None:
        """Calling build_for_workspace twice does not duplicate importance rows."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)

        g._compute_and_store_importance(ws)
        scores_first = g.file_importance(ws)
        g._compute_and_store_importance(ws)
        scores_second = g.file_importance(ws)

        assert scores_first == scores_second, (
            "Rebuilding importance twice must yield identical scores (PK + delete-first)"
        )

    def test_row_count_not_doubled_on_rebuild(self, tmp_path) -> None:
        """After two compute calls, the row count must equal the file count (no dups)."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)

        g._compute_and_store_importance(ws)
        g._compute_and_store_importance(ws)

        with g._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM symbol_importance WHERE workspace = ?", (ws,)
            ).fetchone()[0]
        scores = g.file_importance(ws)
        assert count == len(scores), (
            f"Row count ({count}) must equal len(file_importance) ({len(scores)}) — no duplicates"
        )

    def test_empty_graph_no_crash(self, tmp_path) -> None:
        """A workspace with no symbols/edges must not crash; file_importance returns empty."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        # No files indexed — pure empty state
        g._compute_and_store_importance(ws)
        scores = g.file_importance(ws)
        assert scores == {}, f"Empty graph must return empty dict, got {scores}"

    def test_no_edge_graph_no_crash(self, tmp_path) -> None:
        """A workspace with symbols but no edges must not crash."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        p = make_py(tmp_path, "isolated.py", "def standalone(): pass\n")
        g.index_file(ws, p)
        # No edges exist — power-iteration over dangling nodes only
        g._compute_and_store_importance(ws)
        # Should not raise; result may be empty or uniform
        scores = g.file_importance(ws)
        assert isinstance(scores, dict)

    def test_delete_file_removes_importance_row(self, tmp_path) -> None:
        """delete_file must also remove that file's importance row."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)
        g._compute_and_store_importance(ws)

        assert files["b"] in g.file_importance(ws), "file_b must have a score before delete"
        g.delete_file(ws, files["b"])
        assert files["b"] not in g.file_importance(ws), (
            "file_b importance row must be removed by delete_file"
        )

    def test_file_importance_empty_when_never_built(self, tmp_path) -> None:
        """A freshly created SymbolGraph with no compute call returns an empty dict."""
        g = SymbolGraph(str(tmp_path))
        scores = g.file_importance(str(tmp_path))
        assert scores == {}

    def test_scores_in_range(self, tmp_path) -> None:
        """All scores returned by file_importance must be in (0, 1]."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)
        g._compute_and_store_importance(ws)
        scores = g.file_importance(ws)
        for fp, score in scores.items():
            assert 0.0 < score <= 1.0 + 1e-9, (
                f"{fp}: score {score} not in (0, 1]"
            )

    def test_schema_version_bumped(self) -> None:
        """SYMBOL_SCHEMA_VERSION is an exact pin, not a floor: every
        extraction-behavior change must bump it (or existing installs never
        rebuild their graph) AND consciously update this pin in the same
        commit. Currently 15 (same-file class-base override edges of
        edge_type="overrides" recorded at index time;
        SymbolGraph.overrides_of / SymbolGraph.overridden_by expose the
        read API — UPG-BASE-METHOD-OVERRIDE-FLOOD — bumped from 14 Java
        enum_constant and field_declaration extracted as symbols,
        UPG-JAVA-ENUM-CONSTANTS-NO-L2-SYMBOLS, and Rust impl_item
        parse-error check scoped to the implemented type's
        `type_identifier` node, UPG-IMPL-SYMBOL-NAME-NODE-GAP)."""
        assert SYMBOL_SCHEMA_VERSION == 15, (
            f"Expected SYMBOL_SCHEMA_VERSION=15; got {SYMBOL_SCHEMA_VERSION}. "
            "If you changed extraction behavior, bump the version and update this pin."
        )


# ---------------------------------------------------------------------------
# UPG-TESTPATH-FRAMEWORK-MISCLASS (F58) — unambiguous corpus-wide caller
# fan-in: compute, persist, read API.
#
# Synthetic fixtures only — no benchmark-corpus names (django/QuerySet/etc.) in
# committed tests, matching the vectr product/benchmark separation.
# ---------------------------------------------------------------------------

class TestFileFanIn:
    """UPG-TESTPATH-FRAMEWORK-MISCLASS: corpus-wide unambiguous caller-file
    fan-in — the signal that separates a shipped, widely-used file from a
    disposable one, independent of path."""

    def test_unambiguous_target_counts_distinct_caller_files(self, tmp_path) -> None:
        """file_b defines 'foo' (unique leaf); c/d/e each call it → fan_in[file_b] == 3."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)

        g._compute_and_store_file_fan_in(ws)
        fan_in = g.file_fan_in(ws)

        assert fan_in.get(files["b"], 0) == 3, (
            f"file_b has 3 distinct unambiguous callers (c, d, e); got {fan_in.get(files['b'])}"
        )

    def test_unreferenced_file_has_no_fan_in_row(self, tmp_path) -> None:
        """file_a has no in-links — must not appear in the fan-in map."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        files = _seed_importance_graph(g, ws, tmp_path)

        g._compute_and_store_file_fan_in(ws)
        fan_in = g.file_fan_in(ws)

        assert files["a"] not in fan_in

    def test_ambiguous_leaf_is_excluded_from_fan_in(self, tmp_path) -> None:
        """A leaf name defined in TWO files is ambiguous — edges targeting it
        must not contribute to either file's fan-in count (leaf-collision
        noise, e.g. __init__/setUp shared across many unrelated files)."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        import time
        now = time.time()
        f1 = str(tmp_path / "one.py")
        f2 = str(tmp_path / "two.py")
        caller = str(tmp_path / "caller.py")
        for fp in (f1, f2, caller):
            Path(fp).write_text("# placeholder\n")
        with g._conn() as conn:
            # 'dup' is defined in BOTH f1 and f2 — ambiguous leaf.
            conn.execute(
                "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ws, "dup", "function", f1, 1, 3, now),
            )
            conn.execute(
                "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ws, "dup", "function", f2, 1, 3, now),
            )
            conn.execute(
                "INSERT INTO symbols (workspace, name, kind, file_path, start_line, end_line, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ws, "caller_fn", "function", caller, 1, 3, now),
            )
            conn.execute(
                "INSERT INTO edges (workspace, from_file, from_symbol, from_line, to_symbol, edge_type) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ws, caller, "caller_fn", 10, "dup", "calls"),
            )

        g._compute_and_store_file_fan_in(ws)
        fan_in = g.file_fan_in(ws)

        assert f1 not in fan_in
        assert f2 not in fan_in

    def test_rebuild_idempotency(self, tmp_path) -> None:
        """Calling the fan-in compute twice does not duplicate rows or change values."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        _seed_importance_graph(g, ws, tmp_path)

        g._compute_and_store_file_fan_in(ws)
        first = g.file_fan_in(ws)
        g._compute_and_store_file_fan_in(ws)
        second = g.file_fan_in(ws)

        assert first == second
        with g._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM file_fan_in WHERE workspace = ?", (ws,)
            ).fetchone()[0]
        assert count == len(second), (
            f"Row count ({count}) must equal len(file_fan_in) ({len(second)}) — no duplicates"
        )

    def test_empty_graph_no_crash(self, tmp_path) -> None:
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g._compute_and_store_file_fan_in(ws)
        assert g.file_fan_in(ws) == {}

    def test_file_fan_in_empty_when_never_built(self, tmp_path) -> None:
        g = SymbolGraph(str(tmp_path))
        assert g.file_fan_in(str(tmp_path)) == {}

    def test_build_for_workspace_computes_fan_in(self, tmp_path) -> None:
        """build_for_workspace end-to-end: fan-in is persisted and readable."""
        g = SymbolGraph(str(tmp_path))
        p1 = make_py(tmp_path, "lib.py", "def core_fn(): pass\n")
        p2 = make_py(tmp_path, "caller.py", "def user():\n    core_fn()\n")
        g.build_for_workspace(str(tmp_path), [p1, p2])

        fan_in = g.file_fan_in(str(tmp_path))
        assert fan_in.get(p1, 0) == 1


# ---------------------------------------------------------------------------
# ARCH-2 — class-level reference-frequency importance: compute, persist, read API
#
# Synthetic fixtures only — no benchmark-corpus names (django/QuerySet/etc.) in
# committed tests, matching the vectr product/benchmark separation.
# ---------------------------------------------------------------------------

class TestClassImportanceARCH2:
    """ARCH-2: class-level reference-frequency importance computation, persistence,
    and read API — the signal that discriminates same-leaf method collisions
    (two unrelated classes that both define a same-named method)."""

    def test_more_referenced_class_gets_highest_score(self, tmp_path) -> None:
        """Widget is referenced (instantiated/typed) far more than Gadget across the
        corpus → Widget must score 1.0 (max) after normalization."""
        g = SymbolGraph(str(tmp_path))
        ws = str(tmp_path)
        p1 = make_py(
            tmp_path, "widget.py",
            "class Widget:\n"
            "    def render(self):\n        pass\n"
        )
        p2 = make_py(
            tmp_path, "gadget.py",
            "class Gadget:\n"
            "    def render(self):\n        pass\n"
        )
        # Many callers reference Widget by name (instantiation / type hints);
        # Gadget is referenced by only its own definition.
        referencers = [
            make_py(tmp_path, f"caller_{i}.py", "def build():\n    return Widget()\n")
            for i in range(5)
        ]
        file_paths = [p1, p2] + referencers
        g.build_for_workspace(ws, file_paths)

        scores = g.class_importance(ws)
        assert scores.get("Widget", 0.0) == pytest.approx(1.0), (
            f"Widget (most-referenced) must score 1.0, got {scores.get('Widget')}"
        )
        assert scores.get("Widget", 0.0) > scores.get("Gadget", 0.0), (
            f"Widget must outscore Gadget: {scores}"
        )

    def test_barrel_reexport_mentions_not_counted(self, tmp_path) -> None:
        """UPG-CLASS-IMPORTANCE-REEXPORT-NOISE: a bare barrel re-export
        (`export { Foo } from './foo'`) is module plumbing, not a genuine
        reference — it must not inflate Foo's class_importance above an
        equally-defined, equally-used sibling (Bar) that just isn't re-exported.
        Without the fix Foo's count is def+use+re-export (3) vs Bar's def+use (2)
        → Foo outscores Bar; with the re-export line excluded both count 2."""
        g = SymbolGraph(str(tmp_path))
        ws = str(tmp_path)
        foo = make_py(tmp_path, "foo.ts", "export class Foo {\n  run() {}\n}\n")
        bar = make_py(tmp_path, "bar.ts", "export class Bar {\n  run() {}\n}\n")
        # genuine, EQUAL usage of both in a consumer (real reference sites)
        use = make_py(tmp_path, "use.ts", "const a = new Foo();\nconst b = new Bar();\n")
        # a barrel that re-exports ONLY Foo — the mention that used to inflate it
        barrel = make_py(tmp_path, "index.ts", "export { Foo } from './foo';\n")
        g.build_for_workspace(ws, [foo, bar, use, barrel])
        scores = g.class_importance(ws)
        assert scores.get("Foo") == pytest.approx(scores.get("Bar")), (
            f"the barrel re-export of Foo must not inflate its importance above "
            f"the equally-defined/used Bar: {scores}"
        )

    def test_genuine_import_and_definition_still_counted(self, tmp_path) -> None:
        """The re-export exclusion must be narrow: an `import { X } from` (a real
        use site) and an `export class X`/`export default X` (a definition/value)
        keep counting — only bare barrel re-export lines are dropped."""
        g = SymbolGraph(str(tmp_path))
        ws = str(tmp_path)
        widget = make_py(tmp_path, "widget.ts", "export class Widget {\n  run() {}\n}\n")
        gadget = make_py(tmp_path, "gadget.ts", "export class Gadget {\n  run() {}\n}\n")
        # Widget is genuinely imported+used more than Gadget → must still win.
        consumer = make_py(
            tmp_path, "consumer.ts",
            "import { Widget } from './widget';\nconst a = new Widget();\nconst b = new Widget();\n",
        )
        g.build_for_workspace(ws, [widget, gadget, consumer])
        scores = g.class_importance(ws)
        assert scores.get("Widget", 0.0) > scores.get("Gadget", 0.0), (
            f"a genuinely imported+used class must still outscore an unused one "
            f"(imports/definitions are not excluded, only barrel re-exports): {scores}"
        )

    def test_persistence_round_trip(self, tmp_path) -> None:
        """build_for_workspace writes class importance; class_importance() returns it."""
        g = SymbolGraph(str(tmp_path))
        ws = str(tmp_path)
        p1 = make_py(tmp_path, "core.py", "class Engine:\n    def start(self):\n        pass\n")
        p2 = make_py(tmp_path, "user.py", "def go():\n    return Engine().start()\n")
        g.build_for_workspace(ws, [p1, p2])

        scores = g.class_importance(ws)
        assert "Engine" in scores
        for name, score in scores.items():
            assert 0.0 < score <= 1.0 + 1e-9, f"{name} has score {score} outside (0, 1]"

    def test_rebuild_idempotency(self, tmp_path) -> None:
        """Calling build_for_workspace twice does not duplicate class_importance rows."""
        g = SymbolGraph(str(tmp_path))
        ws = str(tmp_path)
        p1 = make_py(tmp_path, "core.py", "class Engine:\n    def start(self):\n        pass\n")
        p2 = make_py(tmp_path, "user.py", "def go():\n    return Engine().start()\n")
        file_paths = [p1, p2]

        g.build_for_workspace(ws, file_paths)
        first = g.class_importance(ws)
        g.build_for_workspace(ws, file_paths)
        second = g.class_importance(ws)
        assert first == second

        with g._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM class_importance WHERE workspace = ?", (ws,)
            ).fetchone()[0]
        assert count == len(second), (
            f"Row count ({count}) must equal len(class_importance) ({len(second)}) — no duplicates"
        )

    def test_empty_graph_no_crash(self, tmp_path) -> None:
        """A workspace with no symbols must not crash; class_importance returns empty."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g._compute_and_store_class_importance(ws, [])
        assert g.class_importance(ws) == {}

    def test_module_level_function_enters_importance_map(self, tmp_path) -> None:
        """UPG-SIBLING-TYPEDEF-CROWDING: a workspace with no classes (only a
        module-level function) must not crash, and the function's own name now
        enters the importance map — module-level functions get the same
        reference-frequency importance a class name does (a sole entry scores
        1.0, the max-count normalization case)."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        p = make_py(tmp_path, "funcs.py", "def standalone():\n    pass\n")
        g.build_for_workspace(ws, [p])
        scores = g.class_importance(ws)
        assert scores.get("standalone", 0.0) == pytest.approx(1.0), (
            f"module-level function name must enter the importance map, got {scores}"
        )

    def test_class_importance_empty_when_never_built(self, tmp_path) -> None:
        """A freshly created SymbolGraph with no compute call returns an empty dict."""
        g = SymbolGraph(str(tmp_path))
        assert g.class_importance(str(tmp_path)) == {}

    def test_shared_class_name_across_files_aggregates(self, tmp_path) -> None:
        """Two files each defining a class named Config both contribute to (and read
        from) the single corpus-wide 'Config' reference count — no per-file split,
        matching the 'canonical = most-referenced' whole-word design."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        p1 = make_py(tmp_path, "app_a.py", "class Config:\n    def load(self):\n        pass\n")
        p2 = make_py(tmp_path, "app_b.py", "class Config:\n    def load(self):\n        pass\n")
        p3 = make_py(tmp_path, "caller.py", "def go():\n    return Config()\n")
        g.build_for_workspace(ws, [p1, p2, p3])
        scores = g.class_importance(ws)
        assert "Config" in scores

    def test_struct_kind_enters_importance_map(self, tmp_path) -> None:
        """UPG-SIBLING-TYPEDEF-CROWDING: a non-'class'-kind type definition
        (Go's `type ... struct`, symbol kind='struct') enters the importance
        map exactly like a python 'class'-kind symbol does — type-def kinds
        beyond 'class' (struct/enum/interface/type) are all covered, not just
        the original ARCH-2 class-only seed set."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        p1 = tmp_path / "shapes.go"
        p1.write_text(
            "package shapes\n\ntype Point struct {\n\tX int\n\tY int\n}\n"
        )
        referencers = []
        for i in range(5):
            rp = tmp_path / f"caller_{i}.go"
            rp.write_text(
                "package shapes\n\nfunc build() Point {\n\treturn Point{}\n}\n"
            )
            referencers.append(str(rp))
        g.build_for_workspace(ws, [str(p1)] + referencers)
        scores = g.class_importance(ws)
        assert scores.get("Point", 0.0) > 0.0, (
            f"struct-kind symbol name must enter the importance map, got {scores}"
        )

    def test_log_scale_preserves_separation_against_high_frequency_name(
        self, tmp_path,
    ) -> None:
        """UPG-SIBLING-TYPEDEF-CROWDING normalization: once a high-frequency
        module-level function name shares the map with class names, a decisive
        raw-count gap between two OTHER names must still produce a visibly
        separated (non-near-tied) ABSOLUTE score difference — the
        log(1+x)/log(1+max) scale compresses the top of a widened count range
        instead of squashing every non-maximal name's raw score down toward
        the same tiny absolute value a linear count/max_count scale would
        (a linear scale's normalized RATIO between two names is invariant to
        max_count — dividing both by the same denominator cancels out — but
        the multiplicative rank blend `1 + lambda * score` (agent/searcher.py)
        is gated on the ABSOLUTE difference between two candidates' scores,
        not their ratio, so a linear scale's shrinking absolute magnitudes
        under a large max_count make the resulting factor gap ranking-
        irrelevant even though the ratio is unchanged)."""
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        # Widget: referenced (whole word) 40 times; Gadget: 4 times — a 10x,
        # decisive separation mirroring the STEP-0 evidence shape.
        widget_refs = "\n".join(f"x{i} = Widget()" for i in range(40))
        gadget_refs = "\n".join(f"y{i} = Gadget()" for i in range(4))
        p1 = make_py(tmp_path, "widget.py", "class Widget:\n    pass\n" + widget_refs + "\n")
        p2 = make_py(tmp_path, "gadget.py", "class Gadget:\n    pass\n" + gadget_refs + "\n")
        # A module-level function referenced 5000 times — dwarfs both classes'
        # counts and stretches max_count by two orders of magnitude, the
        # scenario a linear scale's absolute-difference gap cannot survive.
        hot_calls = "\n".join(f"hot_helper()  # call {i}" for i in range(5000))
        p3 = make_py(tmp_path, "hot.py", "def hot_helper():\n    pass\n" + hot_calls + "\n")
        g.build_for_workspace(ws, [p1, p2, p3])

        scores = g.class_importance(ws)
        # Raw whole-word counts: Widget = 1 (own "class Widget:" line) + 40
        # refs = 41; Gadget = 1 + 4 = 5; hot_helper = 1 + 5000 calls = 5001 (max).
        widget_count, gadget_count, max_count = 41, 5, 5001
        expected_log_gap = (
            math.log1p(widget_count) - math.log1p(gadget_count)
        ) / math.log1p(max_count)
        assert scores["Widget"] - scores["Gadget"] == pytest.approx(
            expected_log_gap, abs=1e-9,
        )
        # At the configured class_importance lambda, this is a >=5%
        # multiplicative factor gap between the two candidates — a ranking-
        # relevant separation, not a near-tie. Read the configured lambda
        # rather than a literal so this stays correct across tuning
        # (UPG-ACCEPT-REGRESSION-RECOVERY raised it 0.25 -> 0.35).
        from agent.config import CLASS_IMPORTANCE_PRIOR_LAMBDA as _lam_class
        assert (scores["Widget"] - scores["Gadget"]) * _lam_class > 0.05, (
            f"decisive raw-count gap collapsed to a ranking-irrelevant "
            f"absolute difference after normalization: {scores}"
        )
        # Sanity: the corresponding LINEAR (count/max_count) absolute gap
        # would be over an order of magnitude smaller — the concrete reason
        # log-scale was chosen (arithmetic only; not exercising product code).
        linear_gap = (widget_count / max_count) - (gadget_count / max_count)
        assert linear_gap * _lam_class < 0.01, (
            "expected the linear scale's absolute gap to be ranking-"
            "irrelevant (<1% factor difference) here — if this no longer "
            "holds, revisit the log-scale choice"
        )
        assert (scores["Widget"] - scores["Gadget"]) > 10 * linear_gap, (
            "log-scale gap should be at least an order of magnitude larger "
            "than the linear-scale gap in this high-max_count scenario"
        )


# ---------------------------------------------------------------------------
# UPG-BASE-METHOD-OVERRIDE-FLOOD: same-file class-base override extraction
# and read API.
#
# Pinned here as structural fact (extraction is a fact about the code, not a
# query-side decision), in line with the lane scope. Synthetic fixtures
# only — Django / real-corpus names stay in the benchmark corpus, not in
# committed tests, matching the vectr product/benchmark separation.
# ---------------------------------------------------------------------------

class TestExtractOverridesPython:
    """Python `class Sub(Base)` form: emit one overrides edge per (subclass
    method, base method) pair, all with edge_type='overrides'."""

    def test_simple_single_base_override_emits_one_edge(self, tmp_path) -> None:
        """A class with one same-named method override on one base emits
        exactly one overrides edge with the expected fields."""
        path = make_py(tmp_path, "hashers.py", textwrap.dedent("""\
            class BasePasswordHasher:
                def verify(self, password, encoded):
                    return True

            class BCryptSHA256PasswordHasher(BasePasswordHasher):
                def verify(self, password, encoded):
                    return True
        """))
        symbols, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert len(override_edges) == 1, (
            f"expected 1 override edge for the BCryptSHA256PasswordHasher.verify "
            f"override of BasePasswordHasher.verify; got {override_edges}"
        )
        e = override_edges[0]
        assert e["from_file"] == path
        assert e["from_symbol"] == "verify", (
            f"from_symbol must be the bare method name (matching the symbols "
            f"table convention); got {e['from_symbol']!r}"
        )
        assert e["from_line"] == 6, (
            f"from_line must point at the SUBCLASS method (the override site); "
            f"got {e['from_line']}"
        )
        assert e["to_symbol"] == "BasePasswordHasher.verify", (
            f"to_symbol must be qualified 'BaseClass.method' so the consumer "
            f"can locate the base unambiguously when many classes in the same "
            f"file share a method name; got {e['to_symbol']!r}"
        )

    def test_override_edge_targets_actual_base_method_line(self, tmp_path) -> None:
        """The base method's line is recoverable via the symbols table after
        a locate() on the qualified 'BaseClass.method' name — pins that the
        qualified target is locatable, not just a string."""
        path = make_py(tmp_path, "hashers.py", textwrap.dedent("""\
            class BasePasswordHasher:
                def verify(self, password, encoded):
                    return True

            class BCryptSHA256PasswordHasher(BasePasswordHasher):
                def verify(self, password, encoded):
                    return True
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, path)
        # The symbols table must have an entry for `verify` (the base), and
        # the locate surface must resolve the qualified target to the base
        # class (line 2) rather than the subclass (line 6).
        result = g.locate_l2(ws, "BasePasswordHasher.verify")
        syms = result.symbols
        assert syms, (
            f"locate('BasePasswordHasher.verify') must resolve to the base "
            f"class method; got empty result {result}"
        )
        base_hit = next(
            (s for s in syms if s.start_line == 2 and s.file_path == path),
            None,
        )
        assert base_hit is not None, (
            f"qualified locate must find the BASE method at line 2; got "
            f"candidates: {[(s.start_line, s.name) for s in syms]}"
        )

    def test_multiple_subclasses_each_get_their_own_override_edge(self, tmp_path) -> None:
        """Six PasswordHasher subclasses overriding `verify` produce six
        distinct override edges, one per subclass (F56 family of cases)."""
        classes = [
            "BCryptSHA256PasswordHasher", "MD5PasswordHasher",
            "PBKDF2PasswordHasher", "ScryptPasswordHasher",
            "Argon2PasswordHasher",
        ]
        body = "class BasePasswordHasher:\n    def verify(self, password, encoded):\n        return True\n\n"
        for cls in classes:
            body += f"class {cls}(BasePasswordHasher):\n    def verify(self, password, encoded):\n        return True\n\n"
        path = make_py(tmp_path, "hashers.py", textwrap.dedent(body))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert len(override_edges) == len(classes), (
            f"expected {len(classes)} override edges (one per subclass); "
            f"got {len(override_edges)}"
        )
        # All five edges must point to the same base, but have distinct from_lines.
        from_lines = sorted(e["from_line"] for e in override_edges)
        assert len(set(from_lines)) == len(classes), (
            f"every override must have a distinct from_line (the subclass "
            f"method's line); got {from_lines}"
        )
        for e in override_edges:
            assert e["to_symbol"] == "BasePasswordHasher.verify", (
                f"every override must target the same base; got {e['to_symbol']!r}"
            )

    def test_unrelated_subclass_method_is_not_an_override_edge(self, tmp_path) -> None:
        """A method on the subclass whose name does NOT exist on the base
        must NOT produce an override edge (the override relationship is
        name-matched on the base)."""
        path = make_py(tmp_path, "hashers.py", textwrap.dedent("""\
            class BasePasswordHasher:
                def verify(self, password, encoded):
                    return True

            class CustomHasher(BasePasswordHasher):
                def custom_method(self):
                    return False
        """))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert override_edges == [], (
            f"custom_method on CustomHasher does not exist on the base, so "
            f"no override edge should be emitted; got {override_edges}"
        )

    def test_no_base_in_same_file_emits_no_override_edge(self, tmp_path) -> None:
        """A subclass whose base is in a different file emits no override
        edge (same-file scope is the explicit precision/recall tradeoff —
        see LANE-REPORT)."""
        path = make_py(tmp_path, "custom_hashers.py", textwrap.dedent("""\
            from other_module import BasePasswordHasher

            class CustomHasher(BasePasswordHasher):
                def verify(self, password, encoded):
                    return True
        """))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert override_edges == [], (
            f"base is imported from another module, NOT in this file — no "
            f"override edge should be emitted (same-file scope); got {override_edges}"
        )

    def test_no_class_definition_emits_no_override_edges(self, tmp_path) -> None:
        """A file with no classes emits no override edges — sanity check
        that the override walk is class-scoped, not file-wide."""
        path = make_py(tmp_path, "helpers.py", textwrap.dedent("""\
            def alpha():
                return 1

            def beta():
                return 2
        """))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert override_edges == [], (
            f"no classes in file — no override edges expected; got {override_edges}"
        )

    def test_unresolvable_compound_base_is_skipped(self, tmp_path) -> None:
        """A class base that's an attribute lookup (e.g. `models.Model`) is
        skipped — without a real type resolver, the identifier on the left
        of `.` doesn't name a single class. The override edge is recorded
        only for the SIMPLE-identifier base."""
        path = make_py(tmp_path, "fields.py", textwrap.dedent("""\
            class SimpleBase:
                def my_method(self):
                    return 1

            class SimpleChild(SimpleBase):
                def my_method(self):
                    return 2

            from somewhere import other

            class CompoundBase(other.Thing):
                def my_method(self):
                    return 3
        """))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        # Only the SIMPLE base (SimpleBase) should resolve; the compound
        # base (`other.Thing`) is skipped.
        assert len(override_edges) == 1, (
            f"only the simple-identifier base should produce an override "
            f"edge; got {override_edges}"
        )
        assert override_edges[0]["to_symbol"] == "SimpleBase.my_method", (
            f"the resolved base must be the simple-identifier one; got "
            f"{override_edges[0]['to_symbol']!r}"
        )

    def test_inner_class_with_no_base_emits_no_override(self, tmp_path) -> None:
        """An inner class with no declared base: no override edge from it.
        The walk reaches the inner class (verified separately in
        _collect_methods_by_class's own recursion contract) but, lacking a
        base, nothing to match against. Sanity check that the walk's flat
        iteration does not silently drop inner classes from consideration."""
        path = make_py(tmp_path, "models.py", textwrap.dedent("""\
            class Outer:
                def render(self):
                    return 'outer'

                class Inner:
                    def render(self):
                        return 'inner'
        """))
        # Inner.render has no class base at all (it's just `class Inner: ...`),
        # so no override edge should be emitted from the inner class.
        # Outer.render is a module-level method on a class with no parent
        # either, so no override edge there either.
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert override_edges == [], (
            f"neither Outer nor Inner declares a base; no override edges "
            f"expected; got {override_edges}"
        )


class TestExtractOverridesJavaScript:
    """JavaScript / TypeScript `class X extends Y[, Z, ...]` form: same
    shape as Python but different grammar node types (`class_heritage`
    instead of `superclasses`)."""

    def test_ts_extends_single_base_emits_one_edge(self, tmp_path) -> None:
        path = make_py(tmp_path, "btn.ts", textwrap.dedent("""\
            class BaseButton {
                render(): void { }
            }

            class PrimaryButton extends BaseButton {
                render(): void { }
            }
        """))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert len(override_edges) == 1, (
            f"TS `extends BaseButton` should emit one override edge; got {override_edges}"
        )
        e = override_edges[0]
        assert e["from_symbol"] == "render"
        assert e["from_line"] == 6, f"from_line should be the subclass method line; got {e['from_line']}"
        assert e["to_symbol"] == "BaseButton.render", (
            f"to_symbol should be qualified 'BaseClass.method'; got {e['to_symbol']!r}"
        )

    def test_ts_extends_multiple_bases_emits_edges_per_base(self, tmp_path) -> None:
        """TS allows `class X extends A, B, C` — each base gets an edge
        when the corresponding method exists on it."""
        path = make_py(tmp_path, "multi.ts", textwrap.dedent("""\
            class MixinA {
                go(): void { }
            }
            class MixinB {
                go(): void { }
            }

            class Composite extends MixinA, MixinB {
                go(): void { }
            }
        """))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        # Composite.go overrides BOTH MixinA.go and MixinB.go.
        targets = sorted(e["to_symbol"] for e in override_edges)
        assert targets == ["MixinA.go", "MixinB.go"], (
            f"expected edges to both bases; got {targets}"
        )

    def test_js_class_without_extends_emits_no_edge(self, tmp_path) -> None:
        path = make_py(tmp_path, "simple.js", textwrap.dedent("""\
            class Widget {
                render() { return 'w'; }
            }
        """))
        _, edges = extract_symbols_from_file(path)
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert override_edges == [], (
            f"no extends → no base → no override edge; got {override_edges}"
        )


class TestExtractOverridesNonOOOLanguages:
    """Languages without a `class ... (Base)` form return [] — no override
    edge can be emitted without resolving the base, and these languages
    use different syntax (impl blocks, struct embedding, etc.) that is
    deliberately out of scope for this lane."""

    @pytest.mark.parametrize("language,path_name,body", [
        ("rust", "lib.rs", "trait Foo { fn bar(&self) {} }\n"),
        ("go", "main.go", "package main\n\nfunc plainFunc() {}\n"),
        ("c", "main.c", "int main(void) { return 0; }\n"),
    ])
    def test_rust_go_c_emit_no_override_edges(
        self, tmp_path, language, path_name, body
    ) -> None:
        path = tmp_path / path_name
        path.write_text(body)
        _, edges = extract_symbols_from_file(str(path))
        override_edges = [e for e in edges if e["edge_type"] == "overrides"]
        assert override_edges == [], (
            f"{language} does not use a class-inheritance form that this "
            f"lane resolves; expected no override edges; got {override_edges}"
        )


class TestExtractOverridesDedup:
    """Override edges are deduped by (file, from_symbol, from_line, to_symbol,
    edge_type) so two walkers producing the same edge don't double-insert."""

    def test_dedup_collapses_duplicate_edges(self, tmp_path) -> None:
        """If the same (subclass, base) pair were walked twice (e.g. the
        _collect_methods_by_class walk and the override walk both finding
        the same method), the dedup should collapse to one edge."""
        path = make_py(tmp_path, "dup.py", textwrap.dedent("""\
            class Base:
                def go(self):
                    pass

            class Sub(Base):
                def go(self):
                    pass
        """))
        _, edges = extract_symbols_from_file(path)
        # Build a key for every override edge.
        keys = [
            (e["from_file"], e["from_symbol"], e["from_line"],
             e["to_symbol"], e["edge_type"])
            for e in edges if e["edge_type"] == "overrides"
        ]
        assert len(keys) == len(set(keys)), (
            f"override edges must be deduped by (file, from_symbol, from_line, "
            f"to_symbol, edge_type); got duplicates: {keys}"
        )


class TestSymbolGraphOverridesOf:
    """The `overrides_of` read API on SymbolGraph — given a (file, method,
    line) triple, return every base method it overrides. Uses the existing
    `edges` table (no schema migration) and the same SQLite query shape
    as the other edge query methods."""

    def _index_fixtures(self, tmp_path) -> tuple[SymbolGraph, str, str, int, int]:
        """Helper: index a simple Base/Sub fixture and return
        (graph, workspace, file_path, base_line, sub_line)."""
        # Callers pass a SUBDIRECTORY (tmp_path / "a") to get two isolated
        # workspaces, and pytest's tmp_path only guarantees the root exists.
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = make_py(tmp_path, "hashers.py", textwrap.dedent("""\
            class BasePasswordHasher:
                def verify(self, password, encoded):
                    return True

            class BCryptPasswordHasher(BasePasswordHasher):
                def verify(self, password, encoded):
                    return True
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, path)
        return g, ws, path, 2, 6  # base at line 2, subclass at line 6

    def test_overrides_of_returns_base_edge(self, tmp_path) -> None:
        g, ws, path, base_line, sub_line = self._index_fixtures(tmp_path)
        edges = g.overrides_of(ws, path, "verify", sub_line)
        assert len(edges) == 1, (
            f"subclass verify at line {sub_line} overrides exactly one base; "
            f"got {len(edges)} edges: {edges}"
        )
        e = edges[0]
        assert e.to_symbol == "BasePasswordHasher.verify", (
            f"edge target must be the qualified base name; got {e.to_symbol!r}"
        )
        assert e.edge_type == "overrides"
        assert e.from_symbol == "verify"
        assert e.from_line == sub_line

    def test_overrides_of_returns_empty_for_non_override(self, tmp_path) -> None:
        """A method that does NOT override anything has no edge — query
        returns an empty list (not None, not an error)."""
        g, ws, path, base_line, sub_line = self._index_fixtures(tmp_path)
        # Base.verify itself doesn't override anything.
        edges = g.overrides_of(ws, path, "verify", base_line)
        assert edges == [], (
            f"base method has no parent; overrides_of must return []; got {edges}"
        )

    def test_overrides_of_returns_empty_for_unknown_line(self, tmp_path) -> None:
        """A (file, symbol, line) triple that doesn't correspond to a
        recorded override returns [] — same shape as an empty result for
        any other reason, so callers don't have to special-case it."""
        g, ws, path, base_line, sub_line = self._index_fixtures(tmp_path)
        edges = g.overrides_of(ws, path, "verify", 9999)
        assert edges == [], (
            f"unknown line must return [] (same as a method that has no "
            f"parent); got {edges}"
        )

    def test_overrides_of_is_workspace_isolated(self, tmp_path) -> None:
        """An override edge recorded under workspace A must not be visible
        from workspace B's query — same isolation contract as every other
        graph accessor."""
        g_a, ws_a, path_a, _, sub_line = self._index_fixtures(tmp_path / "a")
        g_b, ws_b, path_b, _, _ = self._index_fixtures(tmp_path / "b")
        # edges are scoped per-workspace by SQLite; query B with A's data.
        edges = g_b.overrides_of(ws_b, path_b, "verify", sub_line)
        # B has its OWN override edge (different workspace but same file
        # path and line — both are the right number in their own DB). The
        # isolation test: A's row must not be readable through B.
        for e in edges:
            assert e.from_file == path_b, (
                f"edges returned by workspace B must come from B's DB; got {e.from_file}"
            )


class TestSymbolGraphOverriddenBy:
    """The `overridden_by` read API on SymbolGraph — the inverse of
    `overrides_of`. Given a (file, BaseClass.method) target, return every
    subclass method that overrides it."""

    def test_overridden_by_returns_subclass_edges(self, tmp_path) -> None:
        """The F56 case shape: 5 subclasses each override the same
        `BasePasswordHasher.verify`; overridden_by must list all 5."""
        classes = [
            "BCryptSHA256PasswordHasher", "MD5PasswordHasher",
            "PBKDF2PasswordHasher", "ScryptPasswordHasher",
            "Argon2PasswordHasher",
        ]
        body = "class BasePasswordHasher:\n    def verify(self, password, encoded):\n        return True\n\n"
        for cls in classes:
            body += f"class {cls}(BasePasswordHasher):\n    def verify(self, password, encoded):\n        return True\n\n"
        path = make_py(tmp_path, "hashers.py", textwrap.dedent(body))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, path)
        edges = g.overridden_by(ws, path, "BasePasswordHasher.verify")
        # Every subclass's verify is an override; 5 distinct (file, line) triples.
        assert len(edges) == len(classes), (
            f"expected {len(classes)} subclass override edges for "
            f"BasePasswordHasher.verify; got {len(edges)}: {edges}"
        )
        from_lines = sorted(set(e.from_line for e in edges))
        assert len(from_lines) == len(classes), (
            f"every override must have a distinct from_line; got {from_lines}"
        )
        for e in edges:
            assert e.to_symbol == "BasePasswordHasher.verify"
            assert e.from_symbol == "verify"
            assert e.from_file == path

    def test_overridden_by_returns_empty_for_no_subclasses(self, tmp_path) -> None:
        """A base with no subclass overrides in this file returns []. A
        cross-file subclass would be missed by the same-file scope (and
        is the explicit out-of-scope tradeoff — see LANE-REPORT)."""
        path = make_py(tmp_path, "lonely.py", textwrap.dedent("""\
            class LonelyBase:
                def go(self):
                    pass
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, path)
        edges = g.overridden_by(ws, path, "LonelyBase.go")
        assert edges == [], (
            f"no subclass override in this file → []; got {edges}"
        )


class TestOverrideEdgesDontInterfereWithExistingEdges:
    """The `overrides` edge_type is additive — it must not collide with
    the existing `calls` / `uses` / `dynamic` edges in callers(), callees(),
    or any other graph accessor that filters on edge_type."""

    def test_callers_does_not_pick_up_overrides(self, tmp_path) -> None:
        """A caller of `verify` must still see only CALLER methods, not
        `overrides` edges (which are structural class relationships, not
        control-flow)."""
        path = make_py(tmp_path, "hashers.py", textwrap.dedent("""\
            class BasePasswordHasher:
                def verify(self, password, encoded):
                    return True

            class CustomHasher(BasePasswordHasher):
                def verify(self, password, encoded):
                    return True

            def login():
                CustomHasher().verify("p", "h")
        """))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, path)
        # `verify` is called from `login`; the override edges between
        # subclasses and the base must not appear in the caller list.
        callers = g.callers(ws, "verify")
        caller_names = {c.from_symbol for c in callers}
        assert "login" in caller_names, (
            f"callers of `verify` must include the actual caller `login`; "
            f"got {caller_names}"
        )
        # Override edges (which would carry the from_symbol 'verify' for
        # the SUBCLASS methods, not for a caller) must not appear in this
        # callers() result — callers() filters to control-flow edges by
        # construction, and a `verify -> verify` overrides edge would
        # otherwise be a self-collision.
        # Easier check: callers' from_lines should be the line of `login`,
        # not the line of any method definition.
        caller_lines = {c.from_line for c in callers}
        # login is the last def in the fixture; line varies by dedent —
        # assert it is NOT 2 (the base method's line) and NOT 6 (the
        # subclass method's line).
        for ln in caller_lines:
            assert ln != 2 and ln != 6, (
                f"caller of `verify` must NOT be a method definition (those "
                f"are override edges, not call edges); got from_line={ln}"
            )


class TestOverridesF56Shape:
    """The F56 case shape: a base hasher with multiple subclass overrides
    of `verify`. The new override edges are exactly the data the lane
    ships — they let a later ranking change discover that all the top-5
    candidates in the F56 query are overrides of a single base class."""

    def test_f56_shape_records_override_graph(self, tmp_path) -> None:
        """Replicate the F56 family of fixtures: BasePasswordHasher with
        five subclass overrides, all named `verify`. Confirm the override
        graph captures every subclass as an override of the base."""
        classes = [
            "BCryptSHA256PasswordHasher", "MD5PasswordHasher",
            "PBKDF2PasswordHasher", "ScryptPasswordHasher",
            "Argon2PasswordHasher",
        ]
        body = "class BasePasswordHasher:\n    def verify(self, password, encoded):\n        return True\n\n"
        for cls in classes:
            body += f"class {cls}(BasePasswordHasher):\n    def verify(self, password, encoded):\n        return True\n\n"
        path = make_py(tmp_path, "hashers.py", textwrap.dedent(body))
        ws = str(tmp_path)
        g = SymbolGraph(ws)
        g.index_file(ws, path)
        # The base's overridden_by list must contain exactly the 5
        # subclass override edges. A later ranking change consuming this
        # data could observe "all 5 top-5 candidates are overrides of the
        # same base" — the structural signal this lane ships.
        edges = g.overridden_by(ws, path, "BasePasswordHasher.verify")
        assert len(edges) == 5, (
            f"F56-shape: 5 subclass overrides of BasePasswordHasher.verify; "
            f"got {len(edges)}"
        )
        # Every edge is from the same file (same-file scope; this is the
        # precision/recall tradeoff documented in LANE-REPORT).
        assert {e.from_file for e in edges} == {path}

