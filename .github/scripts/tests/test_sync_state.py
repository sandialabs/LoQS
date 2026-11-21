"""Tests for sync-state.sh, run against throwaway git repositories."""

from pathlib import Path

import pytest
from gitrepo import commit, git, make_repo, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "sync-state.sh"


@pytest.fixture
def repo(tmp_path):
    """A bugfix branch and a main that has synced part of it."""
    return make_repo(tmp_path)


def already_synced(repo):
    return "main", "fix1"


def unsynced_fix(repo):
    return "main", "bugfix"


def conflicting_fix(repo):
    git(repo, "switch", "bugfix")
    commit(repo, "feature1", "other content")
    git(repo, "switch", "main")
    return "main", "bugfix"


@pytest.mark.parametrize(
    "builder, word",
    [
        (already_synced, "nothing"),
        (unsynced_fix, "clean"),
        (conflicting_fix, "conflict"),
    ],
    ids=lambda p: getattr(p, "__name__", str(p)),
)
def test_states(repo, builder, word):
    main_ref, branch_ref = builder(repo)
    result = run_script(SCRIPT, repo, main_ref, branch_ref)
    assert (result.returncode, result.stdout.strip()) == (0, word)


def unknown_ref(repo):
    return "origin/main", "bugfix"


def unrelated_history(repo):
    git(repo, "switch", "--orphan", "orphan")
    commit(repo, "orphan1")
    return "main", "orphan"


@pytest.mark.parametrize("builder", [unknown_ref, unrelated_history])
def test_undecidable_is_an_error(repo, builder):
    main_ref, branch_ref = builder(repo)
    result = run_script(SCRIPT, repo, main_ref, branch_ref)
    assert (result.returncode, result.stdout) == (2, "")
