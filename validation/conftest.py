"""Case parametrization, strict known failures and the verdict table."""

from __future__ import annotations

import os

import pytest

from validation import verdict
from validation.cases import (
    declared_checks,
    generate,
    judge,
    validate_declaration,
)
from validation.declarations import DECLARATIONS

_ALL = {
    name: generate(decl) for name, decl in DECLARATIONS.items()
}  # codepack -> cases
_BY_ID = {c.id: c for cases in _ALL.values() for c in cases}
_REPORTS: dict[str, dict[str, str]] = {}
_SECONDS: dict[str, dict[str, float]] = {}  # codepack -> case -> seconds
_TALLIES: dict[str, dict[str, dict]] = {}  # codepack -> case -> ft tally


def pytest_addoption(parser):
    parser.addoption(
        "--fail-on-known",
        action="store_true",
        default=False,
        help="fail the run when any case passed only as a declared known "
        "failure, or was skipped",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "big: case over 32 qubits, run only on request"
    )
    problems = [
        f"{decl.name}: {p}"
        for decl in DECLARATIONS.values()
        for p in validate_declaration(decl)
    ]
    if problems:
        raise pytest.UsageError(
            "invalid validation declarations:\n" + "\n".join(problems)
        )


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    classes = getattr(metafunc.module, "CASE_CLASSES", ())
    params = [
        pytest.param(
            c,
            id=c.id,
            marks=[pytest.mark.big] if c.big else [],
        )
        for cases in _ALL.values()
        for c in cases
        if c.cls in classes
    ]
    metafunc.parametrize("case", params)


@pytest.fixture
def judged(case):
    """Run `fn(case, decl)` -> {failing check: detail} and judge it.

    An exception counts as a failing `outcome` check.
    """
    decl = DECLARATIONS[case.codepack]

    def run(fn):
        try:
            failing = dict(fn(case, decl))
        except Exception as exc:  # noqa: BLE001 -- any codepack error
            failing = {"outcome": f"{type(exc).__name__}: {exc}"}
        judge(declared_checks(decl, case.id), failing, failing)

    return run


def _is_controller(config) -> bool:
    return not hasattr(config, "workerinput")


def pytest_runtest_logreport(report):
    cid = report.nodeid.rpartition("[")[2].rstrip("]")
    if cid not in _BY_ID:
        return
    pack = _BY_ID[cid].codepack
    seconds = _SECONDS.setdefault(pack, {})
    seconds[cid] = seconds.get(cid, 0.0) + report.duration
    if report.when == "call":
        tally = dict(report.user_properties).get("ft")
        if tally is not None:
            _TALLIES.setdefault(pack, {})[cid] = tally
    seen = _REPORTS.setdefault(pack, {})
    if report.failed:
        seen[cid] = "failed" if report.when == "call" else "errored"
    elif report.skipped:
        seen[cid] = "skipped"
    elif report.when == "call" and cid not in seen:
        seen[cid] = verdict.ACCEPTED


def _summaries() -> dict[str, dict]:
    """The verdict summary of each codepack that reported a case."""
    out = {}
    for name, reported in _REPORTS.items():
        declared = {
            cid: declared_checks(DECLARATIONS[name], cid) for cid in reported
        }
        out[name] = verdict.class_summary(
            declared,
            reported,
            _SECONDS.get(name, {}),
            _TALLIES.get(name, {}),
        )
    return out


def _known_left(summary: dict) -> bool:
    return any(b["known"] or b["failed"] for b in summary.values())


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not config.getoption("--fail-on-known") or not _is_controller(config):
        return
    # A skipped case counts as failed in the table, so it fails here too.
    if session.exitstatus == pytest.ExitCode.OK and any(
        _known_left(s) for s in _summaries().values()
    ):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter):
    config = terminalreporter.config
    if not _is_controller(config):
        return
    summaries = _summaries()
    for name, summary in summaries.items():
        terminalreporter.write_line("")
        for line in verdict.verdict_lines(name, summary):
            terminalreporter.write_line(line)
    if not config.getoption("--fail-on-known"):
        return
    left = [n for n, s in summaries.items() if _known_left(s)]
    if left:
        terminalreporter.write_line("")
        terminalreporter.write_line(
            f"known failures left in {', '.join(left)}: they fail the run "
            "under --fail-on-known"
        )
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            for name, summary in summaries.items():
                fh.write("```\n")
                fh.write("\n".join(verdict.verdict_lines(name, summary)))
                fh.write("\n```\n\n")
