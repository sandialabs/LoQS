"""Tester for loqs.tools.migrate.cli"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from loqs.tools.migrate.cli import build_parser, main

# Fully resolvable: both the rename and the InstructionLabel call rewrite
# confidently, leaving nothing for manual review.
LEGACY_SOURCE = (
    "from loqs.core.recordables.patchdict import PatchDict\n\n"
    'patches = PatchDict({"L0": None})\n'
    'label = InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
)

# The rename still rewrites confidently, but a positional splat call's
# contents can't be statically resolved and is left flagged for manual
# review.
FLAGGED_SOURCE = (
    "from loqs.core.recordables.patchdict import PatchDict\n\n"
    'patches = PatchDict({"L0": None})\n'
    "label = InstructionLabel(*label_tuple)\n"
)


def _notebook_with_cell(source: str) -> str:
    """A minimal, single-code-cell notebook wrapping `source`."""
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook)


@pytest.fixture
def legacy_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(LEGACY_SOURCE, encoding="utf-8")
    return path


@pytest.fixture
def flagged_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(FLAGGED_SOURCE, encoding="utf-8")
    return path


@pytest.fixture
def flagged_ipynb_file(tmp_path):
    path = tmp_path / "sample.ipynb"
    path.write_text(_notebook_with_cell(FLAGGED_SOURCE), encoding="utf-8")
    return path


class TestBuildParser:
    def test_requires_at_least_one_path(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_defaults(self, tmp_path):
        args = build_parser().parse_args([str(tmp_path)])
        assert args.paths == [tmp_path]
        assert args.dry_run is False
        assert args.no_backup is False

    def test_accepts_multiple_paths(self, tmp_path):
        other = tmp_path / "other"
        args = build_parser().parse_args([str(tmp_path), str(other)])
        assert args.paths == [tmp_path, other]


class TestMainAsLibraryCall:
    """Exercises `main()` directly (in-process), for fast, precise
    exit-code/side-effect assertions."""

    def test_dry_run_never_writes_and_reports_nonzero(self, flagged_file):
        before = flagged_file.read_text(encoding="utf-8")
        code = main(["--dry-run", str(flagged_file)])
        assert code == 1  # the positional splat call
        assert flagged_file.read_text(encoding="utf-8") == before

    def test_writes_confident_rewrites(self, flagged_file):
        code = main([str(flagged_file)])
        assert code == 1  # the positional splat call still flagged
        rewritten = flagged_file.read_text(encoding="utf-8")
        assert "PatchLayout" in rewritten
        assert "PatchDict" not in rewritten
        assert "InstructionLabel(*label_tuple)" in rewritten  # left untouched, not silently dropped

    def test_fully_resolves_when_nothing_needs_manual_review(self, legacy_file):
        code = main([str(legacy_file)])
        assert code == 0
        rewritten = legacy_file.read_text(encoding="utf-8")
        assert "increment_by=2" in rewritten

    def test_clean_file_exits_zero(self, tmp_path):
        path = tmp_path / "clean.py"
        path.write_text("x = 1\n", encoding="utf-8")
        assert main(["--dry-run", str(path)]) == 0
        assert main([str(path)]) == 0

    def test_directory_is_walked_for_py_ipynb_and_md_files(self, tmp_path):
        (tmp_path / "a.py").write_text(FLAGGED_SOURCE, encoding="utf-8")
        (tmp_path / "b.ipynb").write_text(_notebook_with_cell(FLAGGED_SOURCE), encoding="utf-8")
        (tmp_path / "c.txt").write_text(FLAGGED_SOURCE, encoding="utf-8")  # ignored, wrong suffix
        code = main(["--dry-run", str(tmp_path)])
        assert code == 1

    def test_no_matching_files_is_an_error(self, tmp_path):
        (tmp_path / "b.txt").write_text("not python\n", encoding="utf-8")
        assert main(["--dry-run", str(tmp_path)]) == 2

    def test_ipynb_dry_run_flags_without_writing(self, flagged_ipynb_file):
        before = flagged_ipynb_file.read_text(encoding="utf-8")
        code = main(["--dry-run", str(flagged_ipynb_file)])
        assert code == 1
        assert flagged_ipynb_file.read_text(encoding="utf-8") == before

    def test_ipynb_rewrites_in_place(self, flagged_ipynb_file):
        code = main([str(flagged_ipynb_file)])
        assert code == 1  # the positional splat call still flagged
        rewritten = json.loads(flagged_ipynb_file.read_text(encoding="utf-8"))
        cell_source = "".join(rewritten["cells"][0]["source"])
        assert "PatchLayout" in cell_source
        assert "PatchDict" not in cell_source

    def test_file_level_error_does_not_abort_the_whole_run(self, tmp_path):
        good = tmp_path / "good.py"
        good.write_text("x = 1\n", encoding="utf-8")
        bad = tmp_path / "bad.py"
        bad.write_text("this is not ( valid python\n", encoding="utf-8")
        code = main(["--dry-run", str(tmp_path)])
        assert code == 2


class TestBackup:
    """Rewriting backs up a file it actually rewrites to `<name>.bak` by
    default, unless `--no-backup`/`--dry-run` is given."""

    def test_backs_up_before_rewriting(self, flagged_file):
        before = flagged_file.read_text(encoding="utf-8")
        backup_file = flagged_file.with_name(flagged_file.name + ".bak")

        code = main([str(flagged_file)])

        assert code == 1  # the positional splat call still flagged
        assert backup_file.exists()
        assert backup_file.read_text(encoding="utf-8") == before
        assert flagged_file.read_text(encoding="utf-8") != before

    def test_no_backup_flag_skips_backup(self, flagged_file):
        backup_file = flagged_file.with_name(flagged_file.name + ".bak")
        main(["--no-backup", str(flagged_file)])
        assert not backup_file.exists()

    def test_dry_run_never_creates_a_backup(self, flagged_file):
        backup_file = flagged_file.with_name(flagged_file.name + ".bak")
        main(["--dry-run", str(flagged_file)])
        assert not backup_file.exists()

    def test_no_backup_for_a_file_that_needs_no_changes(self, tmp_path):
        path = tmp_path / "clean.py"
        path.write_text("x = 1\n", encoding="utf-8")
        backup_file = path.with_name(path.name + ".bak")
        main([str(path)])
        assert not backup_file.exists()

    def test_existing_backup_is_overwritten(self, legacy_file):
        backup_file = legacy_file.with_name(legacy_file.name + ".bak")
        backup_file.write_text("stale content from a previous run\n", encoding="utf-8")
        original = legacy_file.read_text(encoding="utf-8")

        main([str(legacy_file)])

        assert backup_file.read_text(encoding="utf-8") == original


class TestSummaryLine:
    """A run always ends with a one-line summary, even when there's
    nothing to report -- otherwise a scan that finds nothing to do looks
    identical to one that silently failed to look at the right files."""

    def test_clean_file_reports_zero_changed_and_flagged(self, tmp_path, capsys):
        path = tmp_path / "clean.py"
        path.write_text("x = 1\n", encoding="utf-8")

        code = main(["--dry-run", str(path)])

        assert code == 0
        out = capsys.readouterr().out
        assert "1 file scanned: 0 would be rewritten, 0 flagged" in out

    def test_dry_run_reports_would_be_rewritten(self, legacy_file, capsys):
        main(["--dry-run", str(legacy_file)])
        out = capsys.readouterr().out
        assert "1 file scanned: 1 would be rewritten, 0 flagged" in out

    def test_reports_rewritten(self, legacy_file, capsys):
        main([str(legacy_file)])
        out = capsys.readouterr().out
        assert "1 file scanned: 1 rewritten, 0 flagged" in out

    def test_flagged_file_reports_nonzero_flagged_count(self, flagged_file, capsys):
        main(["--dry-run", str(flagged_file)])
        out = capsys.readouterr().out
        assert "1 file scanned: 1 would be rewritten, 1 flagged" in out

    def test_pluralizes_file_count(self, tmp_path, capsys):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")

        main(["--dry-run", str(tmp_path)])

        out = capsys.readouterr().out
        assert "2 files scanned: 0 would be rewritten, 0 flagged" in out


class TestConsoleScriptSubprocess:
    """A real end-to-end smoke test: invokes the actual `loqs-migrate`
    console script installed by `pyproject.toml`'s `[project.scripts]`
    entry, confirming the packaging (not just the library call) works."""

    def test_installed_console_script_runs(self, flagged_file):
        executable = shutil.which("loqs-migrate")
        if executable is None:
            pytest.skip("loqs-migrate console script not installed")
        result = subprocess.run(
            [executable, "--dry-run", str(flagged_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "splat call" in result.stdout

    def test_module_invocation_matches_console_script(self, flagged_file):
        result = subprocess.run(
            [sys.executable, "-m", "loqs.tools.migrate.cli", "--dry-run", str(flagged_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
