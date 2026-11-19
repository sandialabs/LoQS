"""Tests for check-main-leak.sh, run against throwaway git repositories."""

from pathlib import Path

import pytest
from gitrepo import commit, git, make_repo, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "check-main-leak.sh"


@pytest.fixture
def repo(tmp_path):
    """A bugfix branch and a main that has synced part of it."""
    return make_repo(tmp_path)


def from_bugfix_tip(repo):
    git(repo, "switch", "-c", "pr", "bugfix")
    commit(repo, "pr1")


def from_older_bugfix_commit(repo):
    git(repo, "switch", "-c", "pr", "fix1")
    commit(repo, "pr1")


def bugfix_tip_itself(repo):
    git(repo, "switch", "-c", "pr", "bugfix")


def cherry_pick_from_main(repo):
    git(repo, "switch", "-c", "pr", "bugfix")
    git(repo, "cherry-pick", "-x", "main")


def bugfix_merged_in(repo):
    git(repo, "switch", "-c", "pr", "fix1")
    commit(repo, "pr1")
    git(repo, "merge", "--no-ff", "-m", "update", "bugfix")


def from_main(repo):
    git(repo, "switch", "-c", "pr", "main")
    commit(repo, "pr1")


def main_merged_in(repo):
    git(repo, "switch", "-c", "pr", "bugfix")
    commit(repo, "pr1")
    git(repo, "merge", "--no-ff", "-m", "update", "main")


def from_main_right_after_sync(repo):
    git(repo, "switch", "main")
    git(repo, "reset", "--hard", "main~3")
    git(repo, "merge", "--no-ff", "-m", "sync", "bugfix")
    git(repo, "switch", "-c", "pr", "main")


def older_main_with_bugfix_merged_in(repo):
    git(repo, "switch", "-c", "pr", "main~2")
    commit(repo, "pr1")
    git(repo, "switch", "bugfix")
    commit(repo, "fix3")
    git(repo, "switch", "main")
    git(repo, "merge", "--no-ff", "-m", "sync2", "bugfix")
    git(repo, "switch", "pr")
    git(repo, "merge", "--no-ff", "-m", "update", "bugfix")


@pytest.mark.parametrize(
    "builder, expected",
    [
        (from_bugfix_tip, 0),
        (from_older_bugfix_commit, 0),
        (bugfix_tip_itself, 0),
        (cherry_pick_from_main, 0),
        (bugfix_merged_in, 0),
        (from_main, 1),
        (main_merged_in, 1),
        (from_main_right_after_sync, 1),
        (older_main_with_bugfix_merged_in, 1),
    ],
    ids=lambda p: getattr(p, "__name__", str(p)),
)
def test_scenarios(repo, builder, expected):
    builder(repo)
    assert (
        run_script(SCRIPT, repo, "main", "bugfix", "pr").returncode == expected
    )


def no_merge_base(repo):
    git(repo, "switch", "--orphan", "orphan")
    commit(repo, "orphan1")
    return "main", "orphan"


def unknown_ref(repo):
    from_bugfix_tip(repo)
    return "origin/main", "pr"


@pytest.mark.parametrize("builder", [unknown_ref, no_merge_base])
def test_undecidable_is_an_error(repo, builder):
    main_ref, head = builder(repo)
    assert run_script(SCRIPT, repo, main_ref, "bugfix", head).returncode == 2
