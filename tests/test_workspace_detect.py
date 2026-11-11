"""
Tests for integrations/workspace_detect.py.

Covers:
  - find_workspace_root: finds .git directory walking up
  - find_workspace_root: falls back to start_path when no .git
  - find_workspace_root: works when given a file path (uses parent)
  - get_gitignore_patterns: reads patterns from .gitignore
  - get_gitignore_patterns: ignores comment lines and blank lines
  - get_gitignore_patterns: returns empty list when no .gitignore
  - should_index_file: accepts supported languages
  - should_index_file: rejects unsupported extensions
  - should_index_file: rejects files in _ALWAYS_SKIP dirs
  - should_index_file: rejects files matching gitignore patterns
  - should_index_file: accepts files not matching any pattern
  - should_index_file: handles directory-pattern syntax ("dist/")
"""
from __future__ import annotations

from pathlib import Path

import pytest

from integrations.workspace_detect import (
    find_workspace_root,
    get_gitignore_patterns,
    should_index_file,
)

_MIXED_VECTRIGNORE = """\
# mixed entry forms (UPG-EXCLUDE-REGEX)
vendor
*.generated.py
re:legacy/.*
re:(unclosed
"""


# ---------------------------------------------------------------------------
# find_workspace_root
# ---------------------------------------------------------------------------

class TestFindWorkspaceRoot:
    def test_finds_git_in_same_dir(self, tmp_path) -> None:
        (tmp_path / ".git").mkdir()
        result = find_workspace_root(str(tmp_path))
        assert result == str(tmp_path)

    def test_finds_git_in_parent_dir(self, tmp_path) -> None:
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)
        result = find_workspace_root(str(subdir))
        assert result == str(tmp_path)

    def test_finds_git_in_grandparent(self, tmp_path) -> None:
        (tmp_path / ".git").mkdir()
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        result = find_workspace_root(str(deep))
        assert result == str(tmp_path)

    def test_fallback_when_no_git(self, tmp_path) -> None:
        subdir = tmp_path / "project"
        subdir.mkdir()
        result = find_workspace_root(str(subdir))
        assert result == str(subdir)

    def test_works_with_file_path(self, tmp_path) -> None:
        (tmp_path / ".git").mkdir()
        f = tmp_path / "main.py"
        f.write_text("pass")
        result = find_workspace_root(str(f))
        assert result == str(tmp_path)

    def test_file_in_subdir_with_git_at_root(self, tmp_path) -> None:
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "pkg"
        sub.mkdir()
        f = sub / "mod.py"
        f.write_text("pass")
        result = find_workspace_root(str(f))
        assert result == str(tmp_path)


# ---------------------------------------------------------------------------
# get_gitignore_patterns
# ---------------------------------------------------------------------------

class TestGetGitignorePatterns:
    def test_reads_patterns(self, tmp_path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n.env\n")
        patterns = get_gitignore_patterns(str(tmp_path))
        assert "*.pyc" in patterns
        assert "__pycache__/" in patterns
        assert ".env" in patterns

    def test_ignores_comment_lines(self, tmp_path) -> None:
        (tmp_path / ".gitignore").write_text("# this is a comment\n*.pyc\n")
        patterns = get_gitignore_patterns(str(tmp_path))
        assert "# this is a comment" not in patterns
        assert "*.pyc" in patterns

    def test_ignores_blank_lines(self, tmp_path) -> None:
        (tmp_path / ".gitignore").write_text("\n*.pyc\n\n*.log\n")
        patterns = get_gitignore_patterns(str(tmp_path))
        assert "" not in patterns
        assert len(patterns) == 2

    def test_no_gitignore_returns_empty(self, tmp_path) -> None:
        patterns = get_gitignore_patterns(str(tmp_path))
        assert patterns == []

    def test_all_comments_returns_empty(self, tmp_path) -> None:
        (tmp_path / ".gitignore").write_text("# only comments here\n# another comment\n")
        patterns = get_gitignore_patterns(str(tmp_path))
        assert patterns == []

    def test_strips_trailing_whitespace(self, tmp_path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc   \n  dist/  \n")
        patterns = get_gitignore_patterns(str(tmp_path))
        assert "*.pyc" in patterns


# ---------------------------------------------------------------------------
# should_index_file
# ---------------------------------------------------------------------------

class TestShouldIndexFile:
    def test_accepts_python_file(self, tmp_path) -> None:
        f = tmp_path / "app.py"
        f.touch()
        assert should_index_file(str(f), []) is True

    def test_accepts_javascript_file(self, tmp_path) -> None:
        f = tmp_path / "app.js"
        f.touch()
        assert should_index_file(str(f), []) is True

    def test_accepts_typescript_file(self, tmp_path) -> None:
        f = tmp_path / "app.ts"
        f.touch()
        assert should_index_file(str(f), []) is True

    def test_accepts_txt_file(self, tmp_path) -> None:
        # UPG-11.3: .txt files are now indexed as prose (language='txt', doc-prose quality).
        # This is the fix for F2: django docs/howto/custom-model-fields.txt was invisible.
        f = tmp_path / "notes.txt"
        f.touch()
        assert should_index_file(str(f), []) is True

    def test_rejects_log_file(self, tmp_path) -> None:
        # .log is NOT in LANG_BY_EXT — use this as the "unsupported extension" check
        f = tmp_path / "debug.log"
        f.touch()
        assert should_index_file(str(f), []) is False

    def test_accepts_markdown_file(self, tmp_path) -> None:
        f = tmp_path / "README.md"
        f.touch()
        assert should_index_file(str(f), []) is True

    def test_rejects_file_in_node_modules(self, tmp_path) -> None:
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        f = nm / "index.js"
        f.touch()
        assert should_index_file(str(f), []) is False

    def test_rejects_file_in_venv(self, tmp_path) -> None:
        venv = tmp_path / "venv" / "lib"
        venv.mkdir(parents=True)
        f = venv / "utils.py"
        f.touch()
        assert should_index_file(str(f), []) is False

    def test_rejects_file_in_pycache(self, tmp_path) -> None:
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        f = cache / "mod.py"
        f.touch()
        assert should_index_file(str(f), []) is False

    def test_rejects_file_in_dot_venv(self, tmp_path) -> None:
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        f = venv / "utils.py"
        f.touch()
        assert should_index_file(str(f), []) is False

    def test_rejects_file_matching_glob_pattern(self, tmp_path) -> None:
        f = tmp_path / "secret.py"
        f.touch()
        assert should_index_file(str(f), ["secret.py"]) is False

    def test_rejects_file_matching_extension_glob(self, tmp_path) -> None:
        f = tmp_path / "gen_code.py"
        f.touch()
        assert should_index_file(str(f), ["gen_*.py"]) is False

    def test_accepts_file_not_matching_any_pattern(self, tmp_path) -> None:
        f = tmp_path / "service.py"
        f.touch()
        assert should_index_file(str(f), ["*.log", "dist/", "node_modules/"]) is True

    def test_directory_slash_pattern_excludes_files(self, tmp_path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        f = dist / "bundle.py"
        f.touch()
        assert should_index_file(str(f), ["dist/"]) is False

    def test_build_dir_pattern(self, tmp_path) -> None:
        build = tmp_path / "build"
        build.mkdir()
        f = build / "output.py"
        f.touch()
        # build is in _ALWAYS_SKIP
        assert should_index_file(str(f), []) is False

    def test_case_insensitive_extension(self, tmp_path) -> None:
        # Extensions are lowercased before checking
        f = tmp_path / "app.PY"
        f.touch()
        assert should_index_file(str(f), []) is True

    def test_no_extension_file_rejected(self, tmp_path) -> None:
        f = tmp_path / "Makefile"
        f.touch()
        assert should_index_file(str(f), []) is False


# ---------------------------------------------------------------------------
# matches_gitignore_pattern — shared predicate behind should_index_file's
# gitignore check AND agent.watcher.CodeWatcher's live exclusion check
# (UPG-REST-STARVATION diagnostics: verifying/fixing gitignore parity
# between the bulk index walk and the live watcher).
# ---------------------------------------------------------------------------

class TestMatchesGitignorePattern:
    def test_filename_glob_matches(self) -> None:
        from integrations.workspace_detect import matches_gitignore_pattern
        assert matches_gitignore_pattern(Path("/ws/app.log"), ["*.log"]) is True

    def test_filename_glob_no_match(self) -> None:
        from integrations.workspace_detect import matches_gitignore_pattern
        assert matches_gitignore_pattern(Path("/ws/app.py"), ["*.log"]) is False

    def test_directory_only_pattern_matches_by_component(self) -> None:
        from integrations.workspace_detect import matches_gitignore_pattern
        assert matches_gitignore_pattern(Path("/ws/dist/bundle.js"), ["dist/"]) is True

    def test_no_patterns_never_matches(self) -> None:
        from integrations.workspace_detect import matches_gitignore_pattern
        assert matches_gitignore_pattern(Path("/ws/anything.py"), []) is False

    def test_used_identically_by_should_index_file(self, tmp_path) -> None:
        # should_index_file's own gitignore check must agree with the
        # standalone predicate for the same inputs — same underlying call,
        # so this can never drift between should_index_file and any other
        # caller (e.g. the watcher) that also calls matches_gitignore_pattern.
        from integrations.workspace_detect import matches_gitignore_pattern
        f = tmp_path / "build.log"
        f.touch()
        patterns = ["*.log"]
        assert matches_gitignore_pattern(f, patterns) is True
        assert should_index_file(str(f), patterns) is False


class TestGitignoreAbsPathCollision:
    # UPG-GITIGNORE-ABSPATH-COLLISION: gitignore/.vectrignore patterns must be
    # matched against the workspace-RELATIVE path, exactly like excluded dir
    # names and re: regexes above them in should_index_file — a workspace
    # living under an ancestor named like an ignore entry (a checkout under
    # /tmp, a CI job dir under build/) must not have every file excluded by
    # its own absolute prefix.
    def test_workspace_under_ignored_named_ancestor_still_indexed(self, tmp_path) -> None:
        root = tmp_path / "tmp" / "proj"
        (root / "src").mkdir(parents=True)
        f = root / "src" / "app.py"
        f.write_text("x = 1\n", encoding="utf-8")
        assert should_index_file(str(f), ["tmp/"], workspace_root=str(root)) is True

    def test_ignored_dir_below_root_still_excluded(self, tmp_path) -> None:
        root = tmp_path / "tmp" / "proj"
        (root / "tmp").mkdir(parents=True)
        f = root / "tmp" / "fixture.py"
        f.write_text("x = 1\n", encoding="utf-8")
        assert should_index_file(str(f), ["tmp/"], workspace_root=str(root)) is False

    def test_path_scoped_glob_matches_workspace_relative(self, tmp_path) -> None:
        # A path-scoped glob entry (docs/*) now matches the workspace-relative
        # path; against the absolute path it could never match at all.
        root = tmp_path / "tmp" / "proj"
        (root / "docs").mkdir(parents=True)
        f = root / "docs" / "conf.py"
        f.write_text("x = 1\n", encoding="utf-8")
        assert should_index_file(str(f), ["docs/*"], workspace_root=str(root)) is False


# ---------------------------------------------------------------------------
# T19: .vectrignore support
# ---------------------------------------------------------------------------

class TestVectrignore:
    def test_get_vectrignore_dirs_empty_when_no_file(self, tmp_path) -> None:
        from integrations.workspace_detect import get_vectrignore_dirs
        assert get_vectrignore_dirs(str(tmp_path)) == set()

    def test_get_vectrignore_dirs_reads_entries(self, tmp_path) -> None:
        from integrations.workspace_detect import get_vectrignore_dirs
        (tmp_path / ".vectrignore").write_text("vendor\ngenerated\n# comment\n")
        dirs = get_vectrignore_dirs(str(tmp_path))
        assert dirs == {"vendor", "generated"}

    def test_get_vectrignore_dirs_skips_comments(self, tmp_path) -> None:
        from integrations.workspace_detect import get_vectrignore_dirs
        (tmp_path / ".vectrignore").write_text("# this is a comment\nproto-gen\n")
        assert get_vectrignore_dirs(str(tmp_path)) == {"proto-gen"}

    def test_write_vectrignore_creates_file(self, tmp_path) -> None:
        from integrations.workspace_detect import write_vectrignore, get_vectrignore_dirs
        write_vectrignore(str(tmp_path), ["vendor", "generated"])
        dirs = get_vectrignore_dirs(str(tmp_path))
        assert dirs == {"vendor", "generated"}

    def test_write_vectrignore_appends_without_duplicates(self, tmp_path) -> None:
        from integrations.workspace_detect import write_vectrignore, get_vectrignore_dirs
        write_vectrignore(str(tmp_path), ["vendor"])
        write_vectrignore(str(tmp_path), ["vendor", "generated"])  # vendor is duplicate
        dirs = get_vectrignore_dirs(str(tmp_path))
        assert dirs == {"vendor", "generated"}
        content = (tmp_path / ".vectrignore").read_text()
        assert content.count("vendor") == 1

    def test_should_index_file_excludes_vectrignore_dir(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file
        vendor_dir = tmp_path / "vendor"
        vendor_dir.mkdir()
        f = vendor_dir / "lib.py"
        f.touch()
        assert should_index_file(str(f), [], extra_excluded_dirs={"vendor"}) is False

    def test_should_index_file_includes_non_excluded_dir(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file
        src = tmp_path / "src"
        src.mkdir()
        f = src / "main.py"
        f.touch()
        assert should_index_file(str(f), [], extra_excluded_dirs={"vendor"}) is True

    def test_indexer_skips_vectrignore_dirs(self, tmp_path, monkeypatch) -> None:
        """index_workspace() must skip directories listed in .vectrignore."""
        from agent import indexer as idx_module
        from tests.conftest import _DummyEmbedProvider

        monkeypatch.setattr(idx_module, "get_embed_provider", lambda _: _DummyEmbedProvider())
        from agent.indexer import CodeIndexer

        # Create workspace: src/main.py (should index) + vendor/lib.py (should skip)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass\n")
        (tmp_path / "vendor").mkdir()
        (tmp_path / "vendor" / "lib.py").write_text("def vendor_fn(): pass\n")

        # Write .vectrignore excluding vendor
        (tmp_path / ".vectrignore").write_text("vendor\n")

        idx = CodeIndexer(str(tmp_path), db_path=str(tmp_path / "chroma"))
        idx._embed_provider = _DummyEmbedProvider()
        files, chunks = idx.index_workspace()

        # Only src/main.py should be indexed
        indexed = idx.indexed_file_paths
        assert any("main.py" in p for p in indexed)
        assert not any("vendor" in p for p in indexed), (
            "vendor/ must be excluded via .vectrignore"
        )

    def test_vectrignore_with_existing_gitignore(self, tmp_path) -> None:
        from integrations.workspace_detect import get_gitignore_patterns, get_vectrignore_dirs
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
        (tmp_path / ".vectrignore").write_text("vendor\n")
        patterns = get_gitignore_patterns(str(tmp_path))
        dirs = get_vectrignore_dirs(str(tmp_path))
        assert "*.pyc" in patterns
        assert "vendor" in dirs


# ---------------------------------------------------------------------------
# UPG-13.3: .vectrignore file glob patterns (additive over bare dir names)
# ---------------------------------------------------------------------------

class TestVectrignoreFileGlobs:
    def test_get_vectrignore_file_globs_reads_glob_entries(self, tmp_path) -> None:
        from integrations.workspace_detect import get_vectrignore_file_globs
        (tmp_path / ".vectrignore").write_text("*.generated.py\nvendor\n", encoding="utf-8")
        assert get_vectrignore_file_globs(str(tmp_path)) == ["*.generated.py"]

    def test_get_vectrignore_file_globs_empty_when_no_globs(self, tmp_path) -> None:
        from integrations.workspace_detect import get_vectrignore_file_globs
        (tmp_path / ".vectrignore").write_text("vendor\ngenerated\n", encoding="utf-8")
        assert get_vectrignore_file_globs(str(tmp_path)) == []

    def test_get_vectrignore_dirs_excludes_glob_entries(self, tmp_path) -> None:
        # Backward compatibility: glob entries must not leak into the bare
        # dir-name set (they're handled by get_vectrignore_file_globs instead).
        from integrations.workspace_detect import get_vectrignore_dirs
        (tmp_path / ".vectrignore").write_text("*.generated.py\nvendor\n", encoding="utf-8")
        assert get_vectrignore_dirs(str(tmp_path)) == {"vendor"}

    def test_get_vectrignore_dirs_unchanged_with_no_globs(self, tmp_path) -> None:
        # Pre-UPG-13.3 behaviour is untouched when there are no glob entries.
        from integrations.workspace_detect import get_vectrignore_dirs
        (tmp_path / ".vectrignore").write_text("vendor\ngenerated\n", encoding="utf-8")
        assert get_vectrignore_dirs(str(tmp_path)) == {"vendor", "generated"}

    def test_question_mark_and_bracket_treated_as_glob(self, tmp_path) -> None:
        from integrations.workspace_detect import get_vectrignore_file_globs, get_vectrignore_dirs
        (tmp_path / ".vectrignore").write_text("file?.py\n[abc].py\nplain_dir\n", encoding="utf-8")
        assert set(get_vectrignore_file_globs(str(tmp_path))) == {"file?.py", "[abc].py"}
        assert get_vectrignore_dirs(str(tmp_path)) == {"plain_dir"}


# ---------------------------------------------------------------------------
# UPG-13.2: default .vectrignore seeding on fresh workspaces
# ---------------------------------------------------------------------------

class TestWriteDefaultVectrignore:
    def test_writes_default_dirs_when_missing(self, tmp_path) -> None:
        from integrations.workspace_detect import write_default_vectrignore, get_vectrignore_dirs
        import agent.config as cfg

        written = write_default_vectrignore(str(tmp_path))
        assert written is True
        assert (tmp_path / ".vectrignore").exists()
        dirs = get_vectrignore_dirs(str(tmp_path))
        assert set(cfg.WORKSPACE_DEFAULT_VECTRIGNORE_DIRS) <= dirs

    def test_never_overwrites_existing_vectrignore(self, tmp_path) -> None:
        from integrations.workspace_detect import write_default_vectrignore

        (tmp_path / ".vectrignore").write_text("custom_only\n", encoding="utf-8")
        written = write_default_vectrignore(str(tmp_path))
        assert written is False
        content = (tmp_path / ".vectrignore").read_text(encoding="utf-8")
        assert content == "custom_only\n"

    def test_never_overwrites_empty_existing_vectrignore(self, tmp_path) -> None:
        from integrations.workspace_detect import write_default_vectrignore

        (tmp_path / ".vectrignore").write_text("", encoding="utf-8")
        written = write_default_vectrignore(str(tmp_path))
        assert written is False
        assert (tmp_path / ".vectrignore").read_text(encoding="utf-8") == ""

    def test_written_file_has_explanatory_comment_header(self, tmp_path) -> None:
        from integrations.workspace_detect import write_default_vectrignore

        write_default_vectrignore(str(tmp_path))
        content = (tmp_path / ".vectrignore").read_text(encoding="utf-8")
        assert content.startswith("#")

    def test_second_call_is_noop(self, tmp_path) -> None:
        from integrations.workspace_detect import write_default_vectrignore

        write_default_vectrignore(str(tmp_path))
        first = (tmp_path / ".vectrignore").read_text(encoding="utf-8")
        write_default_vectrignore(str(tmp_path))
        second = (tmp_path / ".vectrignore").read_text(encoding="utf-8")
        assert first == second


class TestShouldIndexFileWorkspaceUnderExcludedPrefix:
    """Regression (2026-07-03): a workspace living under an excluded-sounding
    absolute prefix (e.g. repo/tmp/fixture, /tmp/myproject) had EVERY file
    excluded because excluded dir names were matched against the full absolute
    path parts. With workspace_root given, only components below the root count.
    Live symptom: the acceptance fixture indexed 0 of 3212 files after a
    default .vectrignore (containing 'tmp') was seeded into it.
    """

    def test_workspace_under_tmp_prefix_is_indexable(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file
        ws = tmp_path / "tmp" / "fixture-project"
        ws.mkdir(parents=True)
        f = ws / "module.py"
        f.write_text("def fn():\n    return 1\n")
        assert should_index_file(
            str(f), [], extra_excluded_dirs={"tmp"}, workspace_root=str(ws)
        ) is True

    def test_excluded_dir_below_root_still_excluded(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file
        ws = tmp_path / "tmp" / "fixture-project"
        sub = ws / "tmp"
        sub.mkdir(parents=True)
        f = sub / "scratch.py"
        f.write_text("x = 1\n")
        assert should_index_file(
            str(f), [], extra_excluded_dirs={"tmp"}, workspace_root=str(ws)
        ) is False

    def test_no_workspace_root_keeps_old_behavior(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file
        f = tmp_path / "tmp" / "proj" / "a.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1\n")
        assert should_index_file(str(f), [], extra_excluded_dirs={"tmp"}) is False


# ---------------------------------------------------------------------------
# UPG-EXCLUDE-REGEX: `re:<pattern>` .vectrignore entries
# ---------------------------------------------------------------------------

class TestVectrignoreRegex:
    def test_mixed_vectrignore_parses_all_forms(self, tmp_path, caplog) -> None:
        from integrations.workspace_detect import (
            get_vectrignore_dirs, get_vectrignore_file_globs, get_vectrignore_regexes,
        )
        (tmp_path / ".vectrignore").write_text(_MIXED_VECTRIGNORE, encoding="utf-8")

        with caplog.at_level("WARNING"):
            regexes = get_vectrignore_regexes(str(tmp_path))

        assert get_vectrignore_dirs(str(tmp_path)) == {"vendor"}
        assert get_vectrignore_file_globs(str(tmp_path)) == ["*.generated.py"]
        # Only the valid regex compiles; the invalid one is skipped, not raised.
        assert len(regexes) == 1
        assert regexes[0].pattern == "legacy/.*"

    def test_invalid_regex_line_logs_one_warning_naming_the_line(self, tmp_path, caplog) -> None:
        from integrations.workspace_detect import get_vectrignore_regexes
        (tmp_path / ".vectrignore").write_text("re:(unclosed\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            regexes = get_vectrignore_regexes(str(tmp_path))

        assert regexes == []
        assert len(caplog.records) == 1
        assert "re:(unclosed" in caplog.records[0].message

    def test_no_vectrignore_returns_empty(self, tmp_path) -> None:
        from integrations.workspace_detect import get_vectrignore_regexes
        assert get_vectrignore_regexes(str(tmp_path)) == []

    def test_regex_entry_excluded_from_dirs_and_globs(self, tmp_path) -> None:
        # A `re:` line must never leak into get_vectrignore_dirs or
        # get_vectrignore_file_globs even though its pattern text often
        # contains glob metacharacters (*, ?, [) itself.
        from integrations.workspace_detect import get_vectrignore_dirs, get_vectrignore_file_globs
        (tmp_path / ".vectrignore").write_text("re:.*_(backup|old)\\.\\w+$\n", encoding="utf-8")
        assert get_vectrignore_dirs(str(tmp_path)) == set()
        assert get_vectrignore_file_globs(str(tmp_path)) == []


class TestShouldIndexFileRegex:
    def test_path_scoped_regex_excludes_files_under_dir(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file, get_vectrignore_regexes
        (tmp_path / ".vectrignore").write_text("re:legacy/.*\n", encoding="utf-8")
        regexes = get_vectrignore_regexes(str(tmp_path))

        legacy = tmp_path / "legacy"
        legacy.mkdir()
        f = legacy / "handler.py"
        f.touch()
        assert should_index_file(
            str(f), [], workspace_root=str(tmp_path), extra_excluded_regexes=regexes
        ) is False

    def test_name_anchored_regex_excludes_matching_filename(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file, get_vectrignore_regexes
        (tmp_path / ".vectrignore").write_text(r"re:.*_backup\.sql$" + "\n", encoding="utf-8")
        regexes = get_vectrignore_regexes(str(tmp_path))

        f = tmp_path / "schema_backup.sql"
        f.touch()
        assert should_index_file(
            str(f), [], workspace_root=str(tmp_path), extra_excluded_regexes=regexes
        ) is False

    def test_regex_does_not_exclude_non_matching_file(self, tmp_path) -> None:
        from integrations.workspace_detect import should_index_file, get_vectrignore_regexes
        (tmp_path / ".vectrignore").write_text("re:legacy/.*\n", encoding="utf-8")
        regexes = get_vectrignore_regexes(str(tmp_path))

        src = tmp_path / "src"
        src.mkdir()
        f = src / "main.py"
        f.touch()
        assert should_index_file(
            str(f), [], workspace_root=str(tmp_path), extra_excluded_regexes=regexes
        ) is True

    def test_file_outside_workspace_root_does_not_crash(self, tmp_path) -> None:
        # Mirrors the existing `parts` fallback for dir-name exclusion: a
        # file that isn't actually under workspace_root falls back to
        # matching its own full path instead of raising.
        from integrations.workspace_detect import should_index_file
        import re as re_mod

        other = tmp_path / "elsewhere" / "module.py"
        other.parent.mkdir(parents=True)
        other.touch()
        ws = tmp_path / "workspace"
        ws.mkdir()
        regexes = [re_mod.compile("legacy/.*")]
        assert should_index_file(
            str(other), [], workspace_root=str(ws), extra_excluded_regexes=regexes
        ) is True


# ---------------------------------------------------------------------------
# validate_workspace_env (UPG-WORKSPACE-ENV-VALIDATE)
# ---------------------------------------------------------------------------

class TestValidateWorkspaceEnv:
    """A typo'd VECTR_WORKSPACE must fail loudly instead of silently falling
    back to cwd-based detection — see api.py's lifespan() and main.py's
    _resolve_workspace_roots(), the two real call sites."""

    def test_nonexistent_path_raises(self, tmp_path) -> None:
        from integrations.workspace_detect import validate_workspace_env, WorkspaceEnvError

        bad_path = str(tmp_path / "does-not-exist")
        with pytest.raises(WorkspaceEnvError) as excinfo:
            validate_workspace_env(bad_path)
        assert bad_path in str(excinfo.value)

    def test_path_that_is_a_file_not_a_directory_raises(self, tmp_path) -> None:
        from integrations.workspace_detect import validate_workspace_env, WorkspaceEnvError

        a_file = tmp_path / "not-a-dir.txt"
        a_file.write_text("x", encoding="utf-8")
        with pytest.raises(WorkspaceEnvError):
            validate_workspace_env(str(a_file))

    def test_valid_directory_does_not_raise(self, tmp_path) -> None:
        from integrations.workspace_detect import validate_workspace_env

        validate_workspace_env(str(tmp_path))  # must not raise

    def test_empty_raw_path_is_a_no_op(self) -> None:
        """Callers pass the already-defaulted value (e.g. "." when the env
        var is unset) — an empty string is never actually produced by
        os.getenv(..., "."), but validate_workspace_env must not raise on it
        either, since it signals "nothing to validate"."""
        from integrations.workspace_detect import validate_workspace_env

        validate_workspace_env("")  # must not raise

    def test_dot_default_never_raises(self, tmp_path, monkeypatch) -> None:
        """The unset-env default ("." — always the existing cwd) must
        never be rejected, so `VECTR_WORKSPACE` being unset is unaffected."""
        from integrations.workspace_detect import validate_workspace_env

        monkeypatch.chdir(tmp_path)
        validate_workspace_env(".")  # must not raise

    def test_error_message_names_the_bad_path_and_env_var(self, tmp_path) -> None:
        from integrations.workspace_detect import validate_workspace_env, WorkspaceEnvError

        bad_path = str(tmp_path / "ghost")
        with pytest.raises(WorkspaceEnvError) as excinfo:
            validate_workspace_env(bad_path, env_var="VECTR_WORKSPACE")
        msg = str(excinfo.value)
        assert "VECTR_WORKSPACE" in msg
        assert bad_path in msg


# ---------------------------------------------------------------------------
# UPG-RAGAS-WORKTREE-GUARD: chunk-quality matchers must judge the file by its
# workspace-RELATIVE path only — never by ancestors above the workspace root.
# should_index_file() used to pass the absolute path into
# is_vectr_config_file/is_generated_file/is_build_artifact_file, whose
# part-scan then saw every directory between / and the file: a checkout
# living under an ancestor named like a generated-dir/config-dir/build-suffix
# component silently lost its ENTIRE tree.
# ---------------------------------------------------------------------------

class TestChunkQualityMatchersAreWorkspaceRelative:
    def test_generated_ancestor_does_not_exclude_tree(self, tmp_path) -> None:
        """RED on main before the fix: `generated` is in
        agent/chunk_quality._GENERATED_DIR_PARTS, and the absolute path
        handed to is_generated_file() contained it as a path ANCESTOR —
        every file under the workspace was excluded no matter its own name."""
        from integrations.workspace_detect import should_index_file

        ws = tmp_path / "generated" / "myproject"
        src = ws / "src"
        src.mkdir(parents=True)
        f = src / "app.py"
        f.write_text("def app(): pass\n")

        assert should_index_file(str(f), [], workspace_root=str(ws)) is True

    def test_config_dir_ancestor_does_not_exclude_tree(self, tmp_path) -> None:
        """`.cursor`/`.vscode` are _VECTR_CONFIG_DIRS; same ancestor bug,
        different set. A workspace checked out under such a directory (e.g.
        dotfile-style layouts) must still index."""
        from integrations.workspace_detect import should_index_file

        ws = tmp_path / ".vscode" / "myproject"
        src = ws / "lib"
        src.mkdir(parents=True)
        f = src / "core.py"
        f.write_text("def core(): pass\n")

        assert should_index_file(str(f), [], workspace_root=str(ws)) is True

    def test_build_suffix_ancestor_does_not_exclude_tree(self, tmp_path) -> None:
        """`mypkg.egg-info` ends with a configured build-artifact suffix;
        as an ANCESTOR it must not exclude the tree (is_build_artifact_file
        scans every directory component of whatever path it is given)."""
        from integrations.workspace_detect import should_index_file

        ws = tmp_path / "mypkg.egg-info" / "checkout"
        src = ws / "pkg"
        src.mkdir(parents=True)
        f = src / "mod.py"
        f.write_text("def mod(): pass\n")

        assert should_index_file(str(f), [], workspace_root=str(ws)) is True

    def test_generated_dir_INSIDE_workspace_still_excluded(self, tmp_path) -> None:
        """The fix must not blind the matchers to relative components: a
        `generated/` directory BELOW the workspace root is still excluded
        exactly as before."""
        from integrations.workspace_detect import should_index_file

        gen = tmp_path / "generated"
        gen.mkdir()
        f = gen / "proto_gen.py"
        f.write_text("def stub(): pass\n")

        assert should_index_file(str(f), [], workspace_root=str(tmp_path)) is False

    def test_config_file_INSIDE_workspace_still_excluded(self, tmp_path) -> None:
        """Companion invariant for the config-file matcher: basename matching
        within the tree keeps working after the relativization change. Uses
        agents.md deliberately — its extension passes _SUPPORTED_EXTS, so the
        exclusion comes from the matcher itself, not the extension gate."""
        from integrations.workspace_detect import should_index_file

        cfg = tmp_path / "agents.md"
        cfg.write_text("instructions\n")

        assert should_index_file(str(cfg), [], workspace_root=str(tmp_path)) is False
