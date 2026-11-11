"""Unit tests for indexer, searcher, and chunking logic. No real API calls."""
from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_py(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8")
    tmp.write(textwrap.dedent(content))
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# chunk_file tests
# ---------------------------------------------------------------------------

class TestChunkFile:
    def test_python_ast_chunks(self) -> None:
        path = make_temp_py("""
            def hello(name: str) -> str:
                return f"Hello, {name}"

            class Greeter:
                def greet(self) -> None:
                    pass
        """)
        from agent.indexer import chunk_file
        chunks = chunk_file(path)
        assert len(chunks) >= 2
        symbols = {c.symbol_name for c in chunks}
        assert "hello" in symbols or "Greeter" in symbols

    def test_chunk_id_format(self) -> None:
        # Use a non-trivial function body so the chunk is not filtered by is_trivial_chunk
        # (UPG-15.1: 2-line declaration+pass stubs are now dropped as trivial).
        # This test checks the chunk_id format "file:symbol", not stub filtering.
        path = make_temp_py("""
            def foo(x: int) -> int:
                return x + 1
        """)
        from agent.indexer import chunk_file
        chunks = chunk_file(path)
        assert chunks
        assert ":" in chunks[0].chunk_id

    def test_fallback_windows_for_unsupported_ext(self, tmp_path) -> None:
        rb_file = tmp_path / "app.rb"
        rb_file.write_text("\n".join([f"# line {i}" for i in range(250)]))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(rb_file))
        assert len(chunks) >= 2  # should produce overlapping windows
        assert all(c.node_type == "window" for c in chunks)

    def test_empty_file_returns_no_chunks(self, tmp_path) -> None:
        empty = tmp_path / "empty.py"
        empty.write_text("")
        from agent.indexer import chunk_file
        assert chunk_file(str(empty)) == []

    def test_chunk_preserves_start_end_lines(self) -> None:
        path = make_temp_py("""
            def alpha() -> None:
                pass

            def beta() -> None:
                pass
        """)
        from agent.indexer import chunk_file
        chunks = chunk_file(path)
        for c in chunks:
            assert c.start_line <= c.end_line
            assert c.start_line >= 1


# ---------------------------------------------------------------------------
# UPG-IMPL-BLOCK-CHUNK-TRUNCATION: any container-shaped node — not just a
# class — must recurse into its members instead of taking one capped chunk.
# Synthetic fixtures only (no benchmark-corpus source).
# ---------------------------------------------------------------------------

class TestChunkFileContainerRecursion:
    def _rust_impl_block_source(self, n_padding_methods: int) -> str:
        lines = ["struct Widget { value: i32 }", "", "impl Widget {"]
        lines.append("    fn early_method(&self) -> i32 {")
        lines.append("        self.value")
        lines.append("    }")
        lines.append("")
        for i in range(n_padding_methods):
            lines.append(f"    fn padding_method_{i}(&self) -> i32 {{")
            lines.append(f"        self.value + {i}")
            lines.append("    }")
            lines.append("")
        lines.append("    fn late_method(&self) -> i32 {")
        lines.append("        self.value * 2")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def test_rust_impl_block_method_past_150_lines_is_its_own_chunk(self, tmp_path) -> None:
        # 40 padding methods (4 lines each) push the impl block well past the
        # 150-line max_chunk_lines cap before late_method is even reached —
        # before the fix, impl_item never recursed (not in _CLASS_NODE_TYPES),
        # so the whole block became one chunk capped at 150 lines and
        # late_method's text was silently dropped, never embedded or indexed.
        p = tmp_path / "widget.rs"
        p.write_text(self._rust_impl_block_source(n_padding_methods=40))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        late = [c for c in chunks if c.symbol_name == "late_method"]
        assert late, "late_method (past line 150 of the impl block) got no chunk of its own"
        assert "self.value * 2" in late[0].content
        assert late[0].node_type == "function_item"

    def test_rust_impl_block_method_before_150_lines_still_chunks(self, tmp_path) -> None:
        p = tmp_path / "widget.rs"
        p.write_text(self._rust_impl_block_source(n_padding_methods=40))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        early = [c for c in chunks if c.symbol_name == "early_method"]
        assert early
        assert "self.value" in early[0].content
        assert early[0].node_type == "function_item"

    def test_rust_impl_item_chunk_itself_is_header_capped_not_full_body(self, tmp_path) -> None:
        # The impl_item's own chunk becomes a header (capped at class_header_lines)
        # now that it recurses — mirrors what class_definition already did.
        p = tmp_path / "widget.rs"
        p.write_text(self._rust_impl_block_source(n_padding_methods=40))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        impl_chunks = [c for c in chunks if c.node_type == "impl_item"]
        assert len(impl_chunks) == 1
        from agent.config import INDEXING_CLASS_HEADER_LINES
        assert len(impl_chunks[0].content.splitlines()) <= INDEXING_CLASS_HEADER_LINES

    def test_small_rust_impl_block_unaffected(self, tmp_path) -> None:
        # A short impl block (well under any cap) — every method still chunks
        # individually, same as before the fix.
        p = tmp_path / "widget.rs"
        p.write_text(self._rust_impl_block_source(n_padding_methods=0))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))
        names = {c.symbol_name for c in chunks}
        assert {"early_method", "late_method"} <= names

    def test_python_class_chunking_is_unchanged(self) -> None:
        # Regression guard: class_definition was already in the container
        # fast-path (_CLASS_NODE_TYPES) before this fix — behavior for an
        # ordinary class with several methods must stay identical.
        path = make_temp_py("""
            class Greeter:
                def greet(self) -> str:
                    return "hello"

                def farewell(self) -> str:
                    return "bye"
        """)
        from agent.indexer import chunk_file
        chunks = chunk_file(path)
        node_types = {c.node_type for c in chunks}
        names = {c.symbol_name for c in chunks}
        assert "class_definition" in node_types
        assert {"Greeter", "greet", "farewell"} <= names
        greet_chunk = next(c for c in chunks if c.symbol_name == "greet")
        assert "# class: Greeter" in greet_chunk.content

    # -----------------------------------------------------------------------
    # UPG-JAVA-ENUM-METHOD-CHUNK-DEPTH: tree-sitter-java nests an enum's own
    # methods two hops below `enum_body` (enum_body -> enum_body_declarations
    # -> method_declaration) instead of one (a class's methods sit directly
    # under class_body), so the one-hop container check alone always missed
    # them — `_MEMBER_WRAPPER_NODE_TYPES` allows exactly one scoped extra hop
    # through the known `enum_body_declarations` grammar wrapper.
    # -----------------------------------------------------------------------

    def _java_enum_source(self, n_padding_methods: int) -> str:
        lines = ["public enum Planet {", "    MERCURY, VENUS;", ""]
        lines.append("    public double earlyMethod() {")
        lines.append("        return 1.0;")
        lines.append("    }")
        lines.append("")
        for i in range(n_padding_methods):
            lines.append(f"    public double paddingMethod{i}() {{")
            lines.append(f"        return {i}.0;")
            lines.append("    }")
            lines.append("")
        lines.append("    public double lateMethod() {")
        lines.append("        return 99.0;")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def test_java_enum_method_past_150_lines_is_its_own_chunk(self, tmp_path) -> None:
        # 40 padding methods push the enum body well past the 150-line
        # max_chunk_lines cap before lateMethod is even reached — before the
        # fix, enum_declaration never recursed at all (its one-hop container
        # check always failed), so the whole enum became one chunk capped at
        # 150 lines and lateMethod's text was silently dropped.
        p = tmp_path / "Planet.java"
        p.write_text(self._java_enum_source(n_padding_methods=40))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        late = [c for c in chunks if c.symbol_name == "lateMethod"]
        assert late, "lateMethod (past line 150 of the enum body) got no chunk of its own"
        assert "return 99.0" in late[0].content
        assert late[0].node_type == "method_declaration"

    def test_java_enum_method_before_150_lines_still_chunks(self, tmp_path) -> None:
        p = tmp_path / "Planet.java"
        p.write_text(self._java_enum_source(n_padding_methods=40))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        early = [c for c in chunks if c.symbol_name == "earlyMethod"]
        assert early
        assert "return 1.0" in early[0].content
        assert early[0].node_type == "method_declaration"

    def test_java_enum_chunk_itself_is_header_capped_not_full_body(self, tmp_path) -> None:
        # The enum_declaration's own chunk becomes a header (capped at
        # class_header_lines) now that it recurses — mirrors what
        # class_declaration and impl_item already do.
        p = tmp_path / "Planet.java"
        p.write_text(self._java_enum_source(n_padding_methods=40))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        enum_chunks = [c for c in chunks if c.node_type == "enum_declaration"]
        assert len(enum_chunks) == 1
        from agent.config import INDEXING_CLASS_HEADER_LINES
        assert len(enum_chunks[0].content.splitlines()) <= INDEXING_CLASS_HEADER_LINES

    def test_small_java_enum_unaffected(self, tmp_path) -> None:
        # A short enum (well under any cap) — every method still chunks
        # individually, same as before the fix.
        p = tmp_path / "Planet.java"
        p.write_text(self._java_enum_source(n_padding_methods=0))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))
        names = {c.symbol_name for c in chunks}
        assert {"earlyMethod", "lateMethod"} <= names

    def test_java_plain_class_chunking_is_unchanged(self, tmp_path) -> None:
        # Regression guard: a Java class (not enum) with several methods must
        # keep exactly the same one-hop container behavior — this fix only
        # widens what counts as a container for the enum's own body shape.
        p = tmp_path / "Calculator.java"
        p.write_text(textwrap.dedent("""\
            public class Calculator {
                public int add(int a, int b) {
                    return a + b;
                }

                public int subtract(int a, int b) {
                    return a - b;
                }
            }
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))
        node_types = {c.node_type for c in chunks}
        names = {c.symbol_name for c in chunks}
        assert "class_declaration" in node_types
        assert {"add", "subtract"} <= names

    # -----------------------------------------------------------------------
    # UPG-RUST-IMPL-SYMBOL-NAME-TRAIT-VS-TYPE (chunk side): the impl_item's
    # resolved name feeds `class_context`, so with the positional-scan bug
    # every member chunk of a trait impl carried `# class: <trait>` — the
    # wrong type — into its embedding text.
    # -----------------------------------------------------------------------

    def _rust_trait_impl_source(self, n_padding_methods: int) -> str:
        lines = ["struct Widget { value: i32 }", "", "impl Measurable for Widget {"]
        lines.append("    fn early_method(&self) -> i32 {")
        lines.append("        self.value")
        lines.append("    }")
        lines.append("")
        for i in range(n_padding_methods):
            lines.append(f"    fn padding_method_{i}(&self) -> i32 {{")
            lines.append(f"        self.value + {i}")
            lines.append("    }")
            lines.append("")
        lines.append("    fn late_method(&self) -> i32 {")
        lines.append("        self.value * 2")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def test_rust_trait_impl_members_carry_type_class_context(self, tmp_path) -> None:
        p = tmp_path / "widget.rs"
        p.write_text(self._rust_trait_impl_source(n_padding_methods=40))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        impl_chunks = [c for c in chunks if c.node_type == "impl_item"]
        assert impl_chunks and impl_chunks[0].symbol_name == "Widget", (
            f"impl chunk named after the trait, not the type: {impl_chunks[0].symbol_name!r}"
        )
        joined = "\n".join(c.content for c in chunks)
        assert "# class: Widget" in joined
        assert "# class: Measurable" not in joined, (
            "member chunks must never carry the TRAIT as their '# class:' context"
        )

    def test_java_constructor_gets_its_own_definition_chunk(self, tmp_path) -> None:
        # UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE (constructor half): with the
        # bug, constructors existed only as lines inside the capped class
        # chunk; now each is its own definition chunk like the methods are.
        p = tmp_path / "Gauge.java"
        p.write_text(textwrap.dedent("""\
            public class Gauge {
                private int ticks;
                public Gauge() { this.ticks = 0; }
                public Gauge(int t) { this.ticks = t; }
                public int read() { return ticks; }
            }
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        ctor_chunks = [c for c in chunks if c.node_type == "constructor_declaration"]
        assert len(ctor_chunks) == 2, (
            f"expected both constructor overloads chunked, got {[c.node_type for c in chunks]}"
        )
        assert all(c.symbol_name == "Gauge" for c in ctor_chunks)
        assert any("this.ticks = t" in c.content for c in ctor_chunks)

    # -----------------------------------------------------------------------
    # UPG-ENUM-CONTAINER-NONMETHOD-COVERAGE (overflow half): header-capping a
    # container used to DROP whatever past-cap content no member chunk carried
    # (fields, static initializers, enum constants); an over-long LEAF lost
    # its whole tail. Both remainders are now re-emitted as bounded
    # "overflow" continuation chunks with honest sub-spans.
    # -----------------------------------------------------------------------

    def _java_constants_only_enum_source(self, n_constants: int) -> str:
        lines = ["public enum Status {"]
        for i in range(n_constants - 1):
            lines.append(f"    STATUS_CODE_{i},")
        lines.append(f"    STATUS_CODE_{n_constants - 1};")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def test_constants_only_java_enum_tail_is_re_emitted(self, tmp_path) -> None:
        # 160 one-line constants ≈ 163 lines and NO methods, so the enum never
        # counts as a member container and is leaf-capped at max_chunk_lines
        # (150). Before the overflow path, everything past line 150 — including
        # the final constant — was silently dropped from the index.
        p = tmp_path / "Status.java"
        p.write_text(self._java_constants_only_enum_source(160))
        from agent.config import INDEXING_MAX_CHUNK_LINES
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        enum_chunks = [c for c in chunks if c.node_type == "enum_declaration"]
        assert len(enum_chunks) == 1
        # The cap itself is untouched — the enum's own chunk stays bounded.
        assert len(enum_chunks[0].content.splitlines()) <= INDEXING_MAX_CHUNK_LINES

        joined = "\n".join(c.content for c in chunks)
        assert "STATUS_CODE_159;" in joined, "final constant past the cap vanished"
        assert "STATUS_CODE_150" in joined

        overflows = [c for c in chunks if c.node_type == "overflow"]
        assert overflows, "expected overflow continuation chunks past the cap"
        # Honest sub-spans: each continuation piece is a verbatim 1:1 slice of
        # the file lines its recorded span names.
        src_lines = p.read_text().splitlines()
        for oc in overflows:
            assert oc.start_line <= oc.end_line
            assert oc.content.splitlines() == src_lines[oc.start_line - 1:oc.end_line], (
                f"{oc.chunk_id}: content does not match its recorded span"
            )

    def _java_methods_then_fields_class_source(self, n_methods: int, n_fields: int) -> str:
        lines = ["public class Registry {"]
        lines.append("    public int earlyMethod() {")
        lines.append("        return 1;")
        lines.append("    }")
        lines.append("")
        for i in range(n_methods):
            lines.append(f"    public int paddingMethod{i}() {{")
            lines.append(f"        return {i};")
            lines.append("    }")
            lines.append("")
        for j in range(n_fields):
            lines.append(f'    public static final String TAIL_CONSTANT_{j} = "v{j}";')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def test_container_tail_fields_past_header_cap_are_re_emitted(self, tmp_path) -> None:
        # ~45 padding methods push everything past line 40 (class_header cap).
        # The tail constants are field_declarations — not chunk targets — so
        # no member chunk carries them; before the overflow path they were
        # unreachable by search AND absent from the symbol graph.
        p = tmp_path / "Registry.java"
        p.write_text(self._java_methods_then_fields_class_source(45, 30))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        joined = "\n".join(c.content for c in chunks)
        assert 'TAIL_CONSTANT_0 = "v0"' in joined, "first tail constant vanished"
        assert 'TAIL_CONSTANT_29 = "v29"' in joined, "last tail constant vanished"

        # Context inheritance: the tail continues the container's MEMBER set,
        # so each piece leads with the same '# class:' line its member chunks
        # carry. (A LEAF's tail instead inherits the leaf's own context — none
        # at top level — staying a verbatim file slice; see the enum test.)
        overflows = [c for c in chunks if c.node_type == "overflow"]
        assert overflows, "expected overflow continuation chunks past the cap"
        for oc in overflows:
            assert oc.content.startswith("# class: Registry\n"), (
                f"{oc.chunk_id}: overflow piece lost the container context line"
            )

        # No-duplication guarantee of the uncovered-range design: a member's
        # body appears in exactly ONE chunk (its own) — never again inside a
        # continuation piece. Naively re-emitting everything past the cap
        # would duplicate every past-cap method body here.
        dup_hits = [c for c in chunks if "return 44;" in c.content]
        assert len(dup_hits) == 1, (
            f"paddingMethod44's body must appear in exactly one chunk, got {len(dup_hits)}"
        )

    def test_overlong_leaf_function_tail_is_re_emitted(self, tmp_path) -> None:
        # A leaf (no nested members) is capped at max_chunk_lines and nothing
        # else carries its tail — the overflow path closes that loss too,
        # under the same invariant: truncation re-emits, never discards.
        from agent.config import INDEXING_MAX_CHUNK_LINES
        lines = ["def big_fn() -> None:"]
        for i in range(290):
            lines.append(f"    x{i} = {i}")
        lines.append("    tail_marker = 'end'")
        p = tmp_path / "big.py"
        p.write_text("\n".join(lines) + "\n")

        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))

        fn_chunks = [c for c in chunks if c.symbol_name == "big_fn"]
        assert len(fn_chunks) == 1
        assert len(fn_chunks[0].content.splitlines()) <= INDEXING_MAX_CHUNK_LINES

        joined = "\n".join(c.content for c in chunks)
        assert "tail_marker" in joined, "function tail past max_chunk_lines vanished"
        assert any(c.node_type == "overflow" for c in chunks)

    def test_no_overflow_chunks_when_nothing_truncates(self, tmp_path) -> None:
        # Files under every cap produce byte-identical output to before this
        # change — the overflow path only ever fires on actual truncation.
        p = tmp_path / "Small.java"
        p.write_text(textwrap.dedent("""\
            public class Small {
                public int once() { return 1; }
                public int twice() { return 2; }
            }
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(p))
        assert not any(c.node_type == "overflow" for c in chunks)
        names = {c.symbol_name for c in chunks}
        assert {"once", "twice"} <= names

    def test_js_closure_callback_still_does_not_trigger_container_recursion(self, tmp_path) -> None:
        # Guards the exact false-positive class this fix must not reintroduce:
        # an anonymous function passed as a call argument sits several hops
        # below the enclosing function's body (wrapped in an expression
        # statement / call arguments), never as a direct body-field child —
        # unlike enum_body_declarations, "wrapped in a call argument" is not a
        # closed, verified grammar wrapper, so it stays outside
        # `_MEMBER_WRAPPER_NODE_TYPES` and must never get its own chunk, no
        # matter how long the enclosing function is.
        lines = ["function processItems(items) {"]
        for i in range(40):
            lines.append(f"  const padding{i} = {i};")
        lines.append("  items.forEach(function(item) {")
        lines.append("    console.log(item);")
        lines.append("  });")
        lines.append("}")
        js_file = tmp_path / "process.js"
        js_file.write_text("\n".join(lines) + "\n")

        from agent.indexer import chunk_file
        chunks = chunk_file(str(js_file))
        node_types = {c.node_type for c in chunks}
        assert "function_expression" not in node_types
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "processItems"


# ---------------------------------------------------------------------------
# Chunking fallback edge cases — mixed-language, window math
# ---------------------------------------------------------------------------

class TestChunkFileFallbackEdgeCases:
    """Covers files with no tree-sitter grammar and window-size boundary behavior."""

    def test_jsx_file_is_chunked_as_javascript(self, tmp_path) -> None:
        jsx_file = tmp_path / "UserCard.jsx"
        jsx_file.write_text(textwrap.dedent("""\
            import React from 'react';

            function UserCard({ name, email }) {
              return (
                <div className="card">
                  <h2>{name}</h2>
                  <p>{email}</p>
                </div>
              );
            }

            export default UserCard;
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(jsx_file))
        assert len(chunks) >= 1
        assert all(c.language == "javascript" for c in chunks)
        combined = "\n".join(c.content for c in chunks)
        assert "UserCard" in combined

    def test_tsx_file_is_chunked_as_typescript(self, tmp_path) -> None:
        tsx_file = tmp_path / "Button.tsx"
        tsx_file.write_text(textwrap.dedent("""\
            import React from 'react';

            interface ButtonProps {
              label: string;
              onClick: () => void;
            }

            function Button({ label, onClick }: ButtonProps) {
              return <button onClick={onClick}>{label}</button>;
            }

            export default Button;
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(tsx_file))
        assert len(chunks) >= 1
        assert all(c.language == "typescript" for c in chunks)
        combined = "\n".join(c.content for c in chunks)
        assert "Button" in combined

    def test_html_file_uses_window_fallback(self, tmp_path) -> None:
        html_file = tmp_path / "index.html"
        html_file.write_text(textwrap.dedent("""\
            <!DOCTYPE html>
            <html>
            <head><title>App</title></head>
            <body>
              <div id="root"></div>
              <script>
                function initApp() {
                  const root = document.getElementById('root');
                  root.innerHTML = '<h1>Hello</h1>';
                }
                initApp();
              </script>
            </body>
            </html>
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(html_file))
        assert len(chunks) >= 1
        assert all(c.node_type == "window" for c in chunks)
        assert all(c.language == "html" for c in chunks)
        combined = "\n".join(c.content for c in chunks)
        assert "initApp" in combined

    def test_jinja_template_uses_window_fallback(self, tmp_path) -> None:
        jinja_file = tmp_path / "users.jinja"
        jinja_file.write_text(textwrap.dedent("""\
            {% extends "base.html" %}
            {% block content %}
              <ul>
              {% for user in users %}
                <li>{{ user.name }} — {{ user.email }}</li>
              {% endfor %}
              </ul>
            {% endblock %}
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(jinja_file))
        assert len(chunks) >= 1
        assert all(c.node_type == "window" for c in chunks)
        combined = "\n".join(c.content for c in chunks)
        assert "user.name" in combined

    def test_python_inline_sql_preserves_sql_content(self, tmp_path) -> None:
        py_file = tmp_path / "repo.py"
        py_file.write_text(textwrap.dedent("""\
            def get_active_users(db):
                query = '''
                    SELECT id, name, email
                    FROM users
                    WHERE active = TRUE
                    ORDER BY created_at DESC
                '''
                return db.execute(query).fetchall()
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(py_file))
        assert len(chunks) >= 1
        assert all(c.language == "python" for c in chunks)
        combined = "\n".join(c.content for c in chunks)
        assert "SELECT" in combined
        assert "get_active_users" in combined

    def test_window_fallback_small_file_is_single_chunk(self, tmp_path) -> None:
        short_file = tmp_path / "config.rb"
        short_file.write_text("\n".join(f"# line {i}" for i in range(100)))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(short_file))
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 100

    def test_window_fallback_overlap_math(self, tmp_path) -> None:
        # 300-line file: window=200, overlap=50 → step=150
        # Window 1: lines 1-200  (i=0)
        # Window 2: lines 151-300 (i=150)
        long_file = tmp_path / "long.rb"
        long_file.write_text("\n".join(f"# line {i}" for i in range(300)))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(long_file))
        assert len(chunks) == 2
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 200
        assert chunks[1].start_line == 151  # overlap starts here
        assert chunks[1].end_line == 300

    def test_window_fallback_language_label_from_extension(self, tmp_path) -> None:
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text("CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL);")
        from agent.indexer import chunk_file
        chunks = chunk_file(str(sql_file))
        assert len(chunks) >= 1
        assert all(c.language == "sql" for c in chunks)
        assert all(c.node_type == "window" for c in chunks)

    def test_python_module_level_only_uses_window_fallback(self, tmp_path) -> None:
        py_file = tmp_path / "settings.py"
        py_file.write_text(textwrap.dedent("""\
            DEBUG = True
            DATABASE_URL = "postgresql://localhost/mydb"
            SECRET_KEY = "abc123"
            ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
        """))
        from agent.indexer import chunk_file
        chunks = chunk_file(str(py_file))
        assert len(chunks) >= 1
        assert all(c.node_type == "window" for c in chunks)


# ---------------------------------------------------------------------------
# workspace_detect tests
# ---------------------------------------------------------------------------

class TestWorkspaceDetect:
    def test_find_workspace_root_git(self, tmp_path) -> None:
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "app"
        subdir.mkdir(parents=True)
        from integrations.workspace_detect import find_workspace_root
        assert find_workspace_root(str(subdir)) == str(tmp_path)

    def test_find_workspace_root_fallback(self, tmp_path) -> None:
        from integrations.workspace_detect import find_workspace_root
        result = find_workspace_root(str(tmp_path))
        assert result == str(tmp_path)

    def test_should_index_file_py(self, tmp_path) -> None:
        f = tmp_path / "app.py"
        f.touch()
        from integrations.workspace_detect import should_index_file
        assert should_index_file(str(f), []) is True

    def test_should_index_file_gitignore(self, tmp_path) -> None:
        f = tmp_path / "generated.py"
        f.touch()
        from integrations.workspace_detect import should_index_file
        assert should_index_file(str(f), ["generated.py"]) is False

    def test_should_skip_node_modules(self, tmp_path) -> None:
        f = tmp_path / "node_modules" / "pkg" / "index.js"
        f.parent.mkdir(parents=True)
        f.touch()
        from integrations.workspace_detect import should_index_file
        assert should_index_file(str(f), []) is False

    def test_should_skip_unsupported_ext(self, tmp_path) -> None:
        f = tmp_path / "data.csv"
        f.touch()
        from integrations.workspace_detect import should_index_file
        assert should_index_file(str(f), []) is False


# ---------------------------------------------------------------------------
# vscode_bridge tests
# ---------------------------------------------------------------------------

class TestVscodeBridge:
    def test_configure_cursor_creates_file(self, tmp_path) -> None:
        from integrations.vscode_bridge import configure_cursor
        configure_cursor(str(tmp_path), port=8765)
        mcp_file = tmp_path / ".cursor" / "mcp.json"
        assert mcp_file.exists()
        import json
        data = json.loads(mcp_file.read_text())
        assert "vectr" in data["mcpServers"]
        assert "8765" in data["mcpServers"]["vectr"]["url"]

    def test_configure_cursor_merges_existing(self, tmp_path) -> None:
        import json
        mcp_dir = tmp_path / ".cursor"
        mcp_dir.mkdir()
        existing = {"mcpServers": {"other-tool": {"url": "http://localhost:9000"}}}
        (mcp_dir / "mcp.json").write_text(json.dumps(existing))
        from integrations.vscode_bridge import configure_cursor
        configure_cursor(str(tmp_path), port=8765)
        data = json.loads((mcp_dir / "mcp.json").read_text())
        assert "other-tool" in data["mcpServers"]  # existing preserved
        assert "vectr" in data["mcpServers"]  # new added

    def test_configure_claude_code(self, tmp_path) -> None:
        from integrations.vscode_bridge import configure_claude_code
        configure_claude_code(str(tmp_path), port=8765)
        settings = tmp_path / ".claude" / "settings.json"
        assert settings.exists()
        import json
        data = json.loads(settings.read_text())
        assert data["mcpServers"]["vectr"]["type"] == "http"


# ---------------------------------------------------------------------------
# MCP server tests
# ---------------------------------------------------------------------------

class TestMcpServer:
    def test_tools_list_returns_all_tools(self) -> None:
        from integrations.mcp_server import handle_tools_list
        result = handle_tools_list()
        names = {t["name"] for t in result["tools"]}
        # core tools
        assert "vectr_search" in names
        assert "vectr_status" in names
        # L1 map (passport)
        assert "vectr_map" in names
        assert "vectr_map_save" in names
        # L2 symbol graph
        assert "vectr_locate" in names
        assert "vectr_trace" in names
        # memory layer
        assert "vectr_remember" in names
        assert "vectr_recall" in names
        assert "vectr_evict_hint" in names
        assert "vectr_snapshot" in names

    def test_tools_call_search_missing_query(self) -> None:
        from integrations.mcp_server import handle_tools_call
        mock_svc = MagicMock()
        result = handle_tools_call("vectr_search", {}, mock_svc)
        assert result["isError"] is True

    def test_tools_call_status(self) -> None:
        from integrations.mcp_server import handle_tools_call
        mock_svc = MagicMock()
        mock_svc.status.return_value = {
            "indexed_files": 42,
            "total_chunks": 1200,
            "last_indexed": "2026-01-01T00:00:00Z",
            "embed_model": "BAAI/bge-base-en-v1.5",
            "workspace_root": "/repo",
        }
        result = handle_tools_call("vectr_status", {}, mock_svc)
        assert result["isError"] is False
        assert "42" in result["content"][0]["text"]

    def test_tools_call_map_save(self) -> None:
        from integrations.mcp_server import handle_tools_call
        mock_svc = MagicMock()
        mock_svc.memory_only = False  # full mode — map_save must run, not short-circuit
        # UPG-6.2: save_map returns a shaped result; mock the real shape.
        mock_svc.save_map.return_value = {"saved": True, "existing_summary": None}
        result = handle_tools_call("vectr_map_save", {"summary": "Python FastAPI service."}, mock_svc)
        assert result["isError"] is False
        mock_svc.save_map.assert_called_once_with("Python FastAPI service.", overwrite=False)

    def test_tools_call_map_save_missing_summary(self) -> None:
        from integrations.mcp_server import handle_tools_call
        mock_svc = MagicMock()
        mock_svc.memory_only = False  # full mode — map_save must reach the summary check
        result = handle_tools_call("vectr_map_save", {}, mock_svc)
        assert result["isError"] is True

    def test_tools_call_locate(self) -> None:
        from integrations.mcp_server import handle_tools_call
        mock_svc = MagicMock()
        mock_svc.memory_only = False  # full mode — locate must run, not short-circuit
        mock_svc.locate_with_snippets.return_value = []
        mock_svc.format_locate.return_value = "No results."
        result = handle_tools_call("vectr_locate", {"name": "MyClass"}, mock_svc)
        assert result["isError"] is False
        mock_svc.locate_with_snippets.assert_called_once_with("MyClass", limit=10, caller_file=None)

    def test_tools_call_unknown_tool(self) -> None:
        from integrations.mcp_server import handle_tools_call
        result = handle_tools_call("nonexistent_tool", {}, MagicMock())
        assert result["isError"] is True


# ---------------------------------------------------------------------------
# WorkingContextStore tests
# ---------------------------------------------------------------------------

class TestWorkingContextStore:
    def _store(self, tmp_path):
        from agent.working_context_store import WorkingContextStore
        return WorkingContextStore(str(tmp_path))

    def test_remember_and_recall(self, tmp_path) -> None:
        store = self._store(tmp_path)
        note_id = store.remember("/repo", "Working on segment targeting", tags=["seg"], priority="high")
        assert isinstance(note_id, int)
        notes = store.recall("/repo")
        assert len(notes) == 1
        assert notes[0].content == "Working on segment targeting"
        assert notes[0].priority == "high"

    def test_recall_with_query_filter(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.remember("/repo", "segment targeting entry point is EvaluateSegments", tags=["seg"])
        store.remember("/repo", "bid pipeline starts in RequestBid", tags=["bid"])
        notes = store.recall("/repo", query="EvaluateSegments")
        assert len(notes) == 1
        assert "EvaluateSegments" in notes[0].content

    def test_recall_with_tag_filter(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.remember("/repo", "note A", tags=["seg"])
        store.remember("/repo", "note B", tags=["bid"])
        notes = store.recall("/repo", tags=["bid"])
        assert len(notes) == 1
        assert notes[0].content == "note B"

    def test_forget_removes_note(self, tmp_path) -> None:
        store = self._store(tmp_path)
        nid = store.remember("/repo", "to forget")
        removed = store.forget("/repo", nid)
        assert removed is True
        assert store.recall("/repo") == []

    def test_count_notes_returns_zero_for_empty_workspace(self, tmp_path) -> None:
        store = self._store(tmp_path)
        assert store.count_notes("/repo") == 0

    def test_count_notes_increments_on_remember(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.remember("/repo", "note one")
        assert store.count_notes("/repo") == 1
        store.remember("/repo", "note two")
        assert store.count_notes("/repo") == 2

    def test_count_notes_decrements_on_forget(self, tmp_path) -> None:
        store = self._store(tmp_path)
        nid = store.remember("/repo", "to forget")
        assert store.count_notes("/repo") == 1
        store.forget("/repo", nid)
        assert store.count_notes("/repo") == 0

    def test_count_notes_is_workspace_scoped(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.remember("/repo/a", "note in a")
        store.remember("/repo/b", "note in b")
        assert store.count_notes("/repo/a") == 1
        assert store.count_notes("/repo/b") == 1

    def test_eviction_hint_lists_chunks(self, tmp_path) -> None:
        store = self._store(tmp_path)
        chunks = [{"file": "main.py", "lines": "1-40", "symbol": "run", "content": "def run(): pass"}]
        hint = store.build_eviction_hint("/repo", chunks)
        assert "main.py" in hint
        assert "<50ms" in hint

    def test_snapshot_and_restore(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.remember("/repo", "snapshot test note")
        sid = store.snapshot("/repo", label="test-snap")
        restored = store.restore_snapshot(sid)
        assert restored is not None
        assert len(restored["notes"]) == 1


# ---------------------------------------------------------------------------
# SymbolGraph tests
# ---------------------------------------------------------------------------

class TestSymbolGraph:
    def _graph(self, tmp_path):
        from agent.symbol_graph import SymbolGraph
        return SymbolGraph(str(tmp_path))

    def test_extract_python_symbols(self) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        path = make_temp_py("""
            def foo():
                bar()

            class MyClass:
                def method(self):
                    pass
        """)
        symbols, edges = extract_symbols_from_file(path)
        names = {s["name"] for s in symbols}
        assert "foo" in names
        assert "MyClass" in names

    def test_extract_python_call_edges(self) -> None:
        from agent.symbol_graph import extract_symbols_from_file
        path = make_temp_py("""
            def caller():
                callee()
        """)
        symbols, edges = extract_symbols_from_file(path)
        call_names = {e["to_symbol"] for e in edges if e["edge_type"] == "calls"}
        assert "callee" in call_names

    def test_index_file_and_locate(self, tmp_path) -> None:
        graph = self._graph(tmp_path)
        path = make_temp_py("""
            def find_me():
                pass
        """)
        graph.index_file("/repo", path)
        results = graph.locate("/repo", "find_me")
        assert len(results) == 1
        assert results[0].name == "find_me"
        assert results[0].kind == "function"

    def test_locate_partial_match(self, tmp_path) -> None:
        graph = self._graph(tmp_path)
        path = make_temp_py("""
            def process_segment():
                pass
            def process_bid():
                pass
        """)
        graph.index_file("/repo", path)
        results = graph.locate("/repo", "process")
        assert len(results) == 2

    def test_callers_trace(self, tmp_path) -> None:
        graph = self._graph(tmp_path)
        path = make_temp_py("""
            def evaluate():
                pass
            def run():
                evaluate()
        """)
        graph.index_file("/repo", path)
        callers = graph.callers("/repo", "evaluate")
        assert any(e.from_symbol == "run" for e in callers)

    def test_locate_returns_snippet(self, tmp_path) -> None:
        graph = self._graph(tmp_path)
        path = make_temp_py("""
            def documented_fn():
                x = 1
                return x
        """)
        graph.index_file("/repo", path)
        results = graph.locate("/repo", "documented_fn")
        assert len(results) == 1
        assert "documented_fn" in results[0].snippet

    def test_format_locate_includes_snippet(self, tmp_path) -> None:
        graph = self._graph(tmp_path)
        path = make_temp_py("""
            def my_fn():
                pass
        """)
        graph.index_file("/repo", path)
        symbols = graph.locate("/repo", "my_fn")
        text = graph.format_locate_for_llm(symbols, "my_fn")
        assert "my_fn" in text
        assert "def my_fn" in text

    def test_format_locate_no_results(self, tmp_path) -> None:
        graph = self._graph(tmp_path)
        text = graph.format_locate_for_llm([], "missing")
        assert "No symbol" in text


# ---------------------------------------------------------------------------
# EvictionAdvisor tests
# ---------------------------------------------------------------------------

class TestEvictionAdvisor:
    def test_no_eviction_when_empty(self) -> None:
        from agent.eviction_advisor import EvictionAdvisor
        advisor = EvictionAdvisor()
        assert advisor.should_evict() is False
        assert advisor.eviction_hint() == ""

    def test_eviction_hint_content(self) -> None:
        from agent.eviction_advisor import EvictionAdvisor
        advisor = EvictionAdvisor(eviction_threshold_tokens=1)
        advisor.record("main.py", "1-100", "run", "x" * 500)
        hint = advisor.eviction_hint()
        assert "main.py" in hint
        assert "<50ms" in hint

    def test_threshold_triggers_eviction(self) -> None:
        from agent.eviction_advisor import EvictionAdvisor
        advisor = EvictionAdvisor(eviction_threshold_tokens=10)
        advisor.record("big.py", "1-500", "foo", "x" * 500)
        assert advisor.should_evict() is True

    def test_duplicate_chunk_not_recorded_twice(self) -> None:
        from agent.eviction_advisor import EvictionAdvisor
        advisor = EvictionAdvisor()
        advisor.record("a.py", "1-10", "foo", "content")
        advisor.record("a.py", "1-10", "foo", "content")
        assert len(advisor._chunks) == 1

    def test_clear_session(self) -> None:
        from agent.eviction_advisor import EvictionAdvisor
        advisor = EvictionAdvisor()
        advisor.record("a.py", "1-10", "foo", "content")
        advisor.clear_session()
        assert advisor.should_evict() is False


# llm_client tests removed — vectr no longer makes internal LLM calls.
# Intelligence is delegated entirely to the AI editor (Claude Code, Cursor, etc.)
# that has integrated vectr. See upgrade-plan.md core principles.
