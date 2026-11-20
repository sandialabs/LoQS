"""Tests for sync-merge-kind.sh, run against throwaway git repositories."""

import subprocess
from pathlib import Path

import pytest
from gitrepo import ENV, commit, git, make_repo, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "sync-merge-kind.sh"


def rev(repo, ref):
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo,
        env=ENV,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A sync branch with a hand commit on top of the bugfix branch."""
    make_repo(tmp_path)
    git(tmp_path, "switch", "-c", "sync", "bugfix")
    commit(tmp_path, "hand", "by hand")
    git(tmp_path, "switch", "main")
    return tmp_path


def merged(repo):
    git(repo, "merge", "--no-ff", "-m", "merge", "sync")
    return rev(repo, "sync")


def squashed(repo):
    git(repo, "merge", "--squash", "sync")
    git(repo, "commit", "-m", "squash")
    return rev(repo, "sync")


def other_head(repo):
    merged(repo)
    return rev(repo, "bugfix")


@pytest.mark.parametrize(
    "builder, word",
    [
        (merged, "merge-commit"),
        (squashed, "other"),
        (other_head, "other"),
    ],
    ids=lambda p: getattr(p, "__name__", str(p)),
)
def test_kinds(repo, builder, word):
    head = builder(repo)
    result = run_script(SCRIPT, repo, "HEAD", head)
    assert (result.returncode, result.stdout.strip()) == (0, word)


def test_undecidable_is_an_error(repo):
    result = run_script(SCRIPT, repo, "0" * 40, "1" * 40)
    assert (result.returncode, result.stdout) == (2, "")
