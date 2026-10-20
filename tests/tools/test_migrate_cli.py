"""Tester for loqs.tools.migrate.cli"""

import json
import shutil
import subprocess

import pytest

from loqs.tools.migrate.cli import main

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

# A bare "Iz" string literal -- flagged by default, confidently rewritten
# to "Imrz" with --rename-Iz.
IZ_SOURCE = 'instrument_name = "Iz"\n'

# A colliding inst_kwargs["patch_label"] -- flagged by default (never
# silently merged), renamed with --rename-patch-label (still flagged
# afterward, as a reminder to update the matching Instruction by hand).
PATCH_LABEL_SOURCE = 'label = InstructionLabel("Init Something", None, (), {"patch_label": "L0"})\n'


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
def flagged_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(FLAGGED_SOURCE, encoding="utf-8")
    return path


@pytest.fixture
def iz_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(IZ_SOURCE, encoding="utf-8")
    return path


@pytest.fixture
def patch_label_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(PATCH_LABEL_SOURCE, encoding="utf-8")
    return path


class TestMainAsLibraryCall:
    """Exercises `main()` directly (in-process), for fast, precise
    exit-code/side-effect assertions."""

    def test_dry_run_never_writes_and_reports_nonzero(self, flagged_file):
        before = flagged_file.read_text(encoding="utf-8")
        backup_file = flagged_file.with_name(flagged_file.name + ".bak")
        code = main(["--dry-run", str(flagged_file)])
        assert code == 1  # the positional splat call
        assert flagged_file.read_text(encoding="utf-8") == before
        assert not backup_file.exists()

    def test_writes_confident_rewrites(self, flagged_file):
        code = main([str(flagged_file)])
        assert code == 1  # the positional splat call still flagged
        rewritten = flagged_file.read_text(encoding="utf-8")
        assert "PatchLayout" in rewritten
        assert "PatchDict" not in rewritten
        assert (
            "InstructionLabel(*label_tuple)" in rewritten
        )  # left untouched, not silently dropped

    def test_clean_file_exits_zero(self, tmp_path):
        path = tmp_path / "clean.py"
        path.write_text("x = 1\n", encoding="utf-8")
        backup_file = path.with_name(path.name + ".bak")
        assert main(["--dry-run", str(path)]) == 0
        assert main([str(path)]) == 0
        assert not backup_file.exists()

    def test_directory_is_walked_for_py_and_ipynb_files(self, tmp_path):
        (tmp_path / "a.py").write_text(LEGACY_SOURCE, encoding="utf-8")
        (tmp_path / "b.ipynb").write_text(
            _notebook_with_cell(LEGACY_SOURCE), encoding="utf-8"
        )
        (tmp_path / "c.txt").write_text(
            LEGACY_SOURCE, encoding="utf-8"
        )  # ignored, wrong suffix

        main(["--no-backup", str(tmp_path)])

        notebook = json.loads(
            (tmp_path / "b.ipynb").read_text(encoding="utf-8")
        )
        cell_source = "".join(notebook["cells"][0]["source"])
        assert "PatchLayout" in (tmp_path / "a.py").read_text(encoding="utf-8")
        assert "PatchLayout" in cell_source
        assert (tmp_path / "c.txt").read_text(
            encoding="utf-8"
        ) == LEGACY_SOURCE

    def test_no_matching_files_is_an_error(self, tmp_path):
        (tmp_path / "b.txt").write_text("not python\n", encoding="utf-8")
        assert main(["--dry-run", str(tmp_path)]) == 2

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


class TestManualReviewReport:
    def test_flagged_file_reports_a_headed_block(self, flagged_file, capsys):
        main(["--dry-run", str(flagged_file)])
        out = capsys.readouterr().out
        assert str(flagged_file) in out.splitlines()


class TestRenameIzFlag:
    def test_rewrites_and_leaves_nothing_flagged(self, iz_file):
        code = main(["--rename-Iz", str(iz_file)])
        assert code == 0
        assert (
            iz_file.read_text(encoding="utf-8") == 'instrument_name = "Imrz"\n'
        )


class TestRenamePatchLabelFlag:
    def test_renames_the_key_and_still_flags(self, patch_label_file):
        code = main(
            [str(patch_label_file), "--rename-patch-label", "new_patch_label"]
        )
        assert (
            code == 1
        )  # still flagged, as a reminder to update the Instruction
        rewritten = patch_label_file.read_text(encoding="utf-8")
        assert "new_patch_label" in rewritten


class TestFollowupSuggestion:
    """A run that finds something `--rename-Iz`/`--rename-patch-label`
    could have addressed ends with a suggested follow-up invocation."""

    def test_suggests_rename_iz(self, iz_file, capsys):
        main(["--dry-run", str(iz_file)])
        out = capsys.readouterr().out
        assert "--rename-Iz" in out
        assert "--rename-patch-label" not in out


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
