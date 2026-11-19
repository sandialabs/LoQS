"""Git helpers shared by the script tests: throwaway repositories."""

import itertools
import os
import subprocess

ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


# Strictly increasing commit dates fix the order of `merge-base --all`.
CLOCK = itertools.count(1_700_000_000)


def git(repo, *args):
    subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**ENV, "GIT_COMMITTER_DATE": f"{next(CLOCK)} +0000"},
        check=True,
        capture_output=True,
        timeout=60,
    )


def commit(repo, name, text=None):
    (repo / name).write_text(name if text is None else text)
    git(repo, "add", name)
    git(repo, "commit", "-m", name)


def run_script(script, repo, *args):
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=repo,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )


def make_repo(path):
    """A bugfix branch and a main that has synced part of it."""
    git(path, "init", "-b", "main")
    commit(path, "root")
    git(path, "branch", "bugfix")
    git(path, "switch", "bugfix")
    commit(path, "fix1")
    git(path, "tag", "fix1")
    commit(path, "fix2")
    git(path, "switch", "main")
    commit(path, "feature1")
    git(path, "merge", "--no-ff", "-m", "sync", "fix1")
    commit(path, "feature2")
    return path
