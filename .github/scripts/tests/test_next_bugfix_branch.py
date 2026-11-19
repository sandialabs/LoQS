"""Tests for next-bugfix-branch.sh."""

from pathlib import Path

import pytest
from gitrepo import run_script

SCRIPT = Path(__file__).resolve().parents[1] / "next-bugfix-branch.sh"


@pytest.mark.parametrize(
    "tag, branch",
    [
        ("v1.3", "bugfix-release-v1.3.1"),
        ("v1.3.1", "bugfix-release-v1.3.2"),
        ("v1.3.9", "bugfix-release-v1.3.10"),
        ("v1.10", "bugfix-release-v1.10.1"),
        ("v1.10.2", "bugfix-release-v1.10.3"),
        ("v2.0", "bugfix-release-v2.0.1"),
    ],
)
def test_next_branch(tmp_path, tag, branch):
    result = run_script(SCRIPT, tmp_path, tag)
    assert result.returncode == 0
    assert result.stdout == branch + "\n"


@pytest.mark.parametrize(
    "tag",
    ["v1.4rc1", "v1.3.1-rc1", "rtd-1.1", "1.3.1", "v1.3.1.1", "v1.3.010"],
)
def test_not_a_plain_tag(tmp_path, tag):
    result = run_script(SCRIPT, tmp_path, tag)
    assert result.returncode == 1
    assert result.stdout == ""
