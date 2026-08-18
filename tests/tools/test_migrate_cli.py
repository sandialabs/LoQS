"""Tester for loqs.tools.migrate.cli"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("libcst")

from loqs.tools.migrate.cli import build_parser, main

LEGACY_SOURCE = (
    "from loqs.core.recordables.patchdict import PatchDict\n\n"
    'patches = PatchDict({"L0": None})\n'
    'label = InstructionLabel("Increment", "L0", (), {"increment_by": 2})\n'
)


@pytest.fixture
def legacy_file(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(LEGACY_SOURCE, encoding="utf-8")
    return path


class TestBuildParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_source_defaults(self, tmp_path):
        args = build_parser().parse_args(["source", str(tmp_path)])
        assert args.command == "source"
        assert args.paths == [tmp_path]
        assert args.config is None
        assert args.dry_run is False

    def test_check_has_no_dry_run_flag(self, tmp_path):
        args = build_parser().parse_args(["check", str(tmp_path)])
        assert args.command == "check"
        assert not hasattr(args, "dry_run")


class TestMainAsLibraryCall:
    """Exercises `main()` directly (in-process), for fast, precise
    exit-code/side-effect assertions."""

    def test_check_never_writes_and_reports_nonzero(self, legacy_file):
        before = legacy_file.read_text(encoding="utf-8")
        code = main(["check", str(legacy_file)])
        assert code == 1  # the unresolvable InstructionLabel candidate
        assert legacy_file.read_text(encoding="utf-8") == before

    def test_source_dry_run_never_writes(self, legacy_file):
        before = legacy_file.read_text(encoding="utf-8")
        code = main(["source", "--dry-run", str(legacy_file)])
        assert code == 1
        assert legacy_file.read_text(encoding="utf-8") == before

    def test_source_writes_confident_rewrites(self, legacy_file):
        code = main(["source", str(legacy_file)])
        assert code == 1  # the InstructionLabel candidate still unresolved
        rewritten = legacy_file.read_text(encoding="utf-8")
        assert "PatchLayout" in rewritten
        assert "PatchDict" not in rewritten

    def test_clean_file_exits_zero(self, tmp_path):
        path = tmp_path / "clean.py"
        path.write_text("x = 1\n", encoding="utf-8")
        assert main(["check", str(path)]) == 0
        assert main(["source", str(path)]) == 0

    def test_config_resolves_instruction_registry(self, legacy_file, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    str(legacy_file): (
                        "loqs.codepacks.codepack_trivial_counter:create_qec_code"
                    )
                }
            ),
            encoding="utf-8",
        )
        code = main(
            ["source", "--config", str(config_path), str(legacy_file)]
        )
        assert code == 0  # nothing left to flag once Increment resolves
        rewritten = legacy_file.read_text(encoding="utf-8")
        assert "increment_by=2" in rewritten

    def test_directory_is_walked_for_py_and_md_files(self, tmp_path):
        (tmp_path / "a.py").write_text(LEGACY_SOURCE, encoding="utf-8")
        (tmp_path / "b.txt").write_text(LEGACY_SOURCE, encoding="utf-8")  # ignored, wrong suffix
        code = main(["check", str(tmp_path)])
        assert code == 1

    def test_no_matching_files_is_an_error(self, tmp_path):
        (tmp_path / "b.txt").write_text("not python\n", encoding="utf-8")
        assert main(["check", str(tmp_path)]) == 2

    def test_file_level_error_does_not_abort_the_whole_run(self, tmp_path):
        good = tmp_path / "good.py"
        good.write_text("x = 1\n", encoding="utf-8")
        bad = tmp_path / "bad.py"
        bad.write_text("this is not ( valid python\n", encoding="utf-8")
        code = main(["check", str(tmp_path)])
        assert code == 2


class TestConsoleScriptSubprocess:
    """A real end-to-end smoke test: invokes the actual `loqs-migrate`
    console script installed by `pyproject.toml`'s `[project.scripts]`
    entry, confirming the packaging (not just the library call) works."""

    def test_installed_console_script_runs(self, legacy_file):
        executable = shutil.which("loqs-migrate")
        if executable is None:
            pytest.skip("loqs-migrate console script not installed")
        result = subprocess.run(
            [executable, "check", str(legacy_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "InstructionLabel" in result.stdout or "Could not resolve" in (
            result.stdout
        )

    def test_module_invocation_matches_console_script(self, legacy_file):
        result = subprocess.run(
            [sys.executable, "-m", "loqs.tools.migrate.cli", "check", str(legacy_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
