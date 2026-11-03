"""Each codepack's validation workflow covers its files and passes its name."""

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("stim")
pytest.importorskip("pygsti")

from validation.declarations import DECLARATIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
REUSABLE = ".github/workflows/reusable-validation.yml"
GENERATOR = ROOT / "loqs" / "codepacks" / "generator"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _covers(patterns, rel: str) -> bool:
    """Whether a path pattern equals the file or is a directory's `/**`."""
    return any(
        p == rel or (p.endswith("/**") and rel.startswith(p[:-2]))
        for p in patterns
    )


def _caller_paths(name: str) -> list[str]:
    data = _load(WORKFLOWS / f"validation-{name}.yml")
    triggers: Any = data.get("on", data.get(True))
    return triggers["push"]["paths"]


@pytest.mark.parametrize("name", sorted(DECLARATIONS))
def test_caller_covers_codepack(name):
    """A codepack's change runs no validation, or its badge shows another's."""
    caller = f"validation-{name}.yml"
    data = _load(WORKFLOWS / caller)
    job = next(iter(data["jobs"].values()))
    assert job["uses"] == "./" + REUSABLE
    assert job["with"]["codepack"] == name
    assert job["with"].get("self_tests", False) == (name == "trivial_counter")

    push = _caller_paths(name)
    needed = set(DECLARATIONS[name].files)
    needed.add(f"validation/declarations/{name}.py")
    needed.add(f".github/workflows/{caller}")
    needed.add(REUSABLE)
    needed |= {p.relative_to(ROOT).as_posix() for p in GENERATOR.rglob("*.py")}
    assert sorted(f for f in needed if not _covers(push, f)) == []


def test_every_codepack_file_is_covered():
    """A codepack file under loqs/codepacks that no caller's paths cover."""
    covered = set()
    for name in DECLARATIONS:
        covered |= set(_caller_paths(name))
    shipped = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "loqs" / "codepacks").rglob("*.py")
        if p.parent != GENERATOR
    }
    shipped.discard("loqs/codepacks/__init__.py")
    assert sorted(f for f in shipped if not _covers(covered, f)) == []


def test_reusable_command_fails_on_known_and_runs_big():
    """A job without --fail-on-known shows known bugs green; a big run is empty."""
    data = _load(ROOT / REUSABLE)
    steps = next(iter(data["jobs"].values()))["steps"]
    command = next(s["run"] for s in steps if "pytest" in s.get("run", ""))
    assert "--fail-on-known" in command
    assert "${{ inputs.big && 'big or not big' || 'not big' }}" in command
