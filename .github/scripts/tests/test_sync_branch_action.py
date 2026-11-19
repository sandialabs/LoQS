"""Tests for sync-branch-action.sh, run against throwaway git repositories."""

from pathlib import Path

import pytest
from gitrepo import commit, git, make_repo, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "sync-branch-action.sh"


@pytest.fixture
def repo(tmp_path):
    """A bugfix branch and a main that has synced part of it."""
    return make_repo(tmp_path)


def no_branch(repo):
    return "false"


def stale_branch(repo):
    git(repo, "branch", "sync", "fix1~1")
    return "false"


def at_tip(repo):
    git(repo, "branch", "sync", "bugfix")
    return "true"


def tip_plus_main(repo):
    git(repo, "switch", "-c", "sync", "bugfix")
    git(repo, "merge", "--no-ff", "-m", "main into sync", "main")
    return "true"


def behind(repo):
    git(repo, "branch", "sync", "fix1")
    return "true"


def hand_commits(repo):
    git(repo, "switch", "-c", "sync", "fix1")
    commit(repo, "hand", "by hand")
    return "true"


def hand_commits_conflicting_fix(repo):
    hand_commits(repo)
    git(repo, "switch", "bugfix")
    commit(repo, "hand", "in the fix")
    return "true"


@pytest.mark.parametrize(
    "builder, word",
    [
        (no_branch, "set"),
        (stale_branch, "set"),
        (at_tip, "keep"),
        (tip_plus_main, "keep"),
        (behind, "fast-forward"),
        (hand_commits, "merge"),
        (hand_commits_conflicting_fix, "conflict"),
    ],
    ids=lambda p: getattr(p, "__name__", str(p)),
)
def test_actions(repo, builder, word):
    pr_open = builder(repo)
    result = run_script(SCRIPT, repo, "bugfix", "sync", pr_open)
    assert (result.returncode, result.stdout.strip()) == (0, word)


def missing_sync_branch(repo):
    pass


def unrelated_sync_branch(repo):
    git(repo, "switch", "--orphan", "sync")
    commit(repo, "orphan1")


@pytest.mark.parametrize(
    "builder", [missing_sync_branch, unrelated_sync_branch]
)
def test_undecidable_is_an_error(repo, builder):
    builder(repo)
    result = run_script(SCRIPT, repo, "bugfix", "sync", "true")
    assert (result.returncode, result.stdout) == (2, "")
