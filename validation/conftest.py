"""Case parametrization, strict known failures and the record writer."""

from __future__ import annotations

import time

import pytest

from validation import records
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
_START = [time.time()]
_START_FILES: dict[str, dict[str, str]] = {}  # codepack -> digests at start
_START_RENDERING: dict[str, str] = {}  # codepack -> fingerprint at start


def pytest_addoption(parser):
    parser.addoption(
        "--write-records",
        action="store_true",
        default=False,
        help="write validation/records/<codepack>.json for fully passing "
        "codepacks at session end",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "big: case of a big (many-qubit) config"
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


def pytest_sessionstart(session):
    _START[0] = time.time()
    if session.config.getoption("--write-records"):
        for name, decl in DECLARATIONS.items():
            _START_FILES[name] = records.file_digests(decl)
            _START_RENDERING[name] = records.rendering_fingerprint(decl)


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
        seen[cid] = records.ACCEPTED


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not config.getoption("--write-records") or not _is_controller(config):
        return
    meta = records.run_metadata()
    meta["command"] = records.command_line(config.invocation_params.args)
    meta["runtime_s"] = round(time.time() - _START[0], 1)
    for name, decl in DECLARATIONS.items():
        if name not in _REPORTS:
            continue
        big = [c.name for c in decl.configs if c.big]
        result = records.decide_record(
            decl,
            [c.id for c in _ALL[name]],
            _REPORTS[name],
            _START_FILES[name],
            meta,
            big,
            _START_RENDERING[name],
            _SECONDS.get(name, {}),
            _TALLIES.get(name, {}),
        )
        if isinstance(result, str):
            print(f"\nno record for {name}: {result}")
        else:
            print(f"\nwrote {records.write_record(name, result)}")


def pytest_terminal_summary(terminalreporter):
    if not _is_controller(terminalreporter.config):
        return
    for name, reported in _REPORTS.items():
        declared = {
            cid: declared_checks(DECLARATIONS[name], cid) for cid in reported
        }
        summary = records.class_summary(
            declared,
            reported,
            _SECONDS.get(name, {}),
            _TALLIES.get(name, {}),
        )
        terminalreporter.write_line("")
        for line in records.verdict_lines(name, summary):
            terminalreporter.write_line(line)
