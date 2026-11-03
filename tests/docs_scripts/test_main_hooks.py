"""Tester for docs_scripts.main_hooks"""

from docs_scripts.main_hooks import docs_branch


def test_docs_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("READTHEDOCS_VERSION_TYPE", "branch")
    monkeypatch.setenv("READTHEDOCS_GIT_IDENTIFIER", "feat-x")
    assert docs_branch(str(tmp_path)) == "feat-x"

    # A pull-request build names the PR number, which is not a branch.
    monkeypatch.setenv("READTHEDOCS_VERSION_TYPE", "external")
    monkeypatch.setenv("READTHEDOCS_GIT_IDENTIFIER", "170")
    assert docs_branch(str(tmp_path)) != "170"
