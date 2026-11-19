"""Tests for closing-issues.sh."""

from pathlib import Path

import pytest
from gitrepo import run_script

SCRIPT = Path(__file__).resolve().parents[1] / "closing-issues.sh"

KEYWORDS = [
    "close",
    "closes",
    "closed",
    "fix",
    "FIXES",
    "fixed",
    "resolve",
    "resolves:",
    "resolved",
]

BODIES = {
    "plain": (
        (
            "Closes #154. Closes #74. Closes #118.\r\n"
            "Also fixes CI's flakiness, see "
            "[#154](https://github.com/sandialabs/LoQS/issues/154).\r\n"
        ),
        ["154", "74", "118"],
    ),
    "every_keyword": (
        "\n".join(f"{word} #{i}" for i, word in enumerate(KEYWORDS, 1)),
        [str(i) for i in range(1, 10)],
    ),
    "template_comment": (
        "<!--\nCloses #123. Closes #124.\nPart of #123.\n-->\n"
        "Closes #7\n<!-- Fixes #9 -->\n",
        ["7"],
    ),
    "repeated_and_unclosed": (
        "Closes #5 and fixes #5\n<!-- Fixes #9\n",
        ["5"],
    ),
    "not_closing": (
        "Closes #\nPart of #12.\nSee #13\nFixes sandialabs/pyGSTi#500\n"
        "prefixes #3\nCloses #12abc\n",
        [],
    ),
}


@pytest.mark.parametrize("body, expected", BODIES.values(), ids=list(BODIES))
def test_closing_issues(tmp_path, body, expected):
    result = run_script(SCRIPT, tmp_path, stdin=body)
    assert result.returncode == 0
    assert result.stdout.split() == expected
