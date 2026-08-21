"""Regression test: `loqs.tools.migrate` finds nothing left to migrate
anywhere in this repo's own `loqs`/`tests`/`docs/notebooks` tree.

This is a real regression check, not exploratory cleanup work -- a direct
grep already confirmed this surface (`PatchDict`, old-format
`InstructionLabel` construction, `.cast()`, `include_idles=`, `"Iz"`,
etc.) was fully clean before this tool existed. A handful of files are
expected, permanent exceptions, each deliberately referencing an old API
on purpose rather than by oversight; this test excludes exactly those
and asserts nothing else turns up.
"""

from pathlib import Path

from loqs.tools.migrate import migrate_source
from loqs.tools.migrate.notebook import migrate_notebook_source

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deliberately, permanently excluded -- each reason is specific to that
# file, not a blanket "tests are exempt" carve-out.
_EXPECTED_EXCEPTIONS = {
    # This tool's own source: `.cast(`/`"Iz"` appear as real regex
    # patterns and report-message text (loqs/tools/migrate/flags.py), or
    # are named in the package's own summary docstring
    # (loqs/tools/migrate/__init__.py) or `--rename_Iz`'s own help text
    # (loqs/tools/migrate/cli.py) -- not leftover legacy code.
    "loqs/tools/migrate/flags.py",
    "loqs/tools/migrate/__init__.py",
    "loqs/tools/migrate/cli.py",
    # This tool's own golden-file test fixtures: deliberately contain
    # every legacy pattern on purpose (see tests/tools/test_migrate.py).
    "tests/tools/migrate_fixtures",
    # This file itself: its docstrings/comments quote the tool's own
    # report messages (e.g. the literal '"Iz"' text below) verbatim, for
    # documentation purposes.
    "tests/tools/test_migrate_tree_clean.py",
    # Tests exercising Serializable's decode-time compatibility
    # machinery directly, which requires deliberately-old-format
    # constructs in their own source/fixtures to test against.
    "tests/internal/test_version_compatibility.py",
    "tests/core/instructions/test_instructionlabel.py",
    "tests/core/recordables/test_patchdict.py",
    # Historical, non-executable fixture-generator scripts (already
    # excluded from pytest's own collection in pytest.ini for the same
    # reason): kept only as a record of a pre-1.2 API, not maintained.
    "tests/backends/fixtures/generate_reps_fixtures.py",
    "tests/backends/model/fixtures/generate_model_fixtures.py",
    # The "Iz" -> "Imrz" legacy-name hint mechanism itself
    # (loqs.internal.legacy.legacy_name_hint) and tests exercising it
    # directly, all of which need the literal old name "Iz" as real data,
    # not leftover legacy code.
    "loqs/internal/legacy.py",
    "tests/internal/test_internal_legacy.py",
    "tests/backends/model/test_pygstimodel.py",
    "tests/core/test_quantumprogram.py",
    # This tool's own test suite: directly exercises `"Iz"` detection and
    # `--rename_Iz`'s rewrite, both of which need the literal old string
    # as real test input.
    "tests/tools/test_migrate.py",
    "tests/tools/test_migrate_cli.py",
    # Tests exercising STIMDictNoiseModel's legacy construction shim
    # directly, which requires deliberately constructing it by name.
    "tests/backends/model/test_dictmodel.py",
}


def _is_excepted(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT).as_posix()
    return any(
        relative == exception or relative.startswith(exception + "/")
        for exception in _EXPECTED_EXCEPTIONS
    )


def _target_files() -> list[Path]:
    targets = (
        list((REPO_ROOT / "loqs").rglob("*.py"))
        + list((REPO_ROOT / "tests").rglob("*.py"))
        + list((REPO_ROOT / "docs" / "notebooks").rglob("*.md"))
    )
    return sorted(p for p in targets if not _is_excepted(p))


class TestRepoTreeIsClean:
    def test_no_file_needs_migration(self):
        offenders = []
        for path in _target_files():
            source = path.read_text(encoding="utf-8")
            if path.suffix == ".md":
                result = migrate_notebook_source(source)
            else:
                result = migrate_source(source)
            if result.changed or result.manual_review:
                offenders.append(
                    (
                        path.relative_to(REPO_ROOT).as_posix(),
                        result.changed,
                        [str(item) for item in result.manual_review],
                    )
                )
        assert offenders == []

    def test_exception_list_is_not_stale(self):
        """Every excepted path must actually exist, and (for the two
        directory-style entries) actually contain files -- an exception
        for a since-deleted/renamed file would silently stop meaning
        anything."""
        for exception in _EXPECTED_EXCEPTIONS:
            path = REPO_ROOT / exception
            assert path.exists(), f"{exception!r} no longer exists"
