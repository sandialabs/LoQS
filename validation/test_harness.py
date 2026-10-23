"""Self-tests of the harness: record decision and judging."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from validation import conftest, engine
from validation.cases import (
    declared_checks,
    generate,
    judge,
    validate_declaration,
)
from validation.mutations import MUTATIONS
from validation.test_fault_tolerance import run_ft
from validation.test_patch_independence import run_independence
from validation.test_tracking import run_tracking
from validation.test_truth_tables import run_truth
from validation.declarations import DECLARATIONS
from validation.logical import Cond, Gate, Meas, Prep, check_shot
from validation.records import decide_record, file_digests

DECL = DECLARATIONS["trivial_counter"]
IDS = ["a.default.static.x.1", "a.default.static.x.2"]


FILES = file_digests(DECL)


def _decide(reports, ids=IDS, start_files=FILES):
    return decide_record(DECL, ids, reports, start_files)


def test_record_decision():
    ok = {i: "passed" for i in IDS}
    record = _decide(ok)
    assert isinstance(record, dict)
    assert record["cases"] == {i: "pass" for i in IDS}
    assert set(record["files"]) >= set(DECL.files)
    # A missing case, or one that failed, errored or was skipped, refuses.
    assert isinstance(_decide({IDS[0]: "passed"}), str)
    for outcome in ("failed", "errored", "skipped"):
        assert isinstance(_decide({**ok, IDS[1]: outcome}), str)
    # A file edited during the run refuses.
    edited = {**FILES, DECL.files[0]: "0" * 64}
    assert isinstance(_decide(ok, start_files=edited), str)


def test_report_status(monkeypatch):
    # Phase reports in order: (when, outcome) per case.
    phases = [
        [("setup", "passed"), ("call", "passed"), ("teardown", "passed")],
        [("setup", "passed"), ("call", "failed"), ("teardown", "passed")],
        [("setup", "failed"), ("teardown", "passed")],
        [("setup", "passed"), ("call", "passed"), ("teardown", "failed")],
        [("setup", "skipped"), ("teardown", "passed")],
    ]
    ids = [f"a.default.static.x.{i}" for i in range(len(phases))]
    monkeypatch.setattr(conftest, "_REPORTS", {})
    by_id = {i: SimpleNamespace(codepack="a") for i in ids}
    monkeypatch.setattr(conftest, "_BY_ID", by_id)
    for cid, reports in zip(ids, phases):
        for when, outcome in reports:
            conftest.pytest_runtest_logreport(
                SimpleNamespace(
                    nodeid=f"validation/test_x.py::test_x[{cid}]",
                    when=when,
                    failed=outcome == "failed",
                    skipped=outcome == "skipped",
                )
            )
    expected = ["passed", "failed", "errored", "errored", "skipped"]
    assert conftest._REPORTS == {"a": dict(zip(ids, expected))}


CID = generate(DECL)[0].id


@pytest.mark.parametrize(
    "known, reproducers",
    [
        ({f"{CID}:O9": "steane-1"}, {}),
        ({"nomatch.*:outcome": "steane-1"}, {}),
        ({f"{CID}:O1": "no-such-bug"}, {}),
        ({}, {"r": ("no.such.case", ("outcome",))}),
        ({f"{CID}:outcome": "steane-1"}, {"r": (CID, ("outcome",))}),
    ],
)
def test_validate_declaration(known, reproducers):
    valid = replace(
        DECL,
        known_failures={f"{CID}:O1": "steane-1"},
        reproducers={"r": (CID, ("outcome",))},
    )
    assert validate_declaration(valid) == []
    bad = replace(DECL, known_failures=known, reproducers=reproducers)
    assert len(validate_declaration(bad)) == 1


def test_check_shot():
    prep_h = [Prep("a", "+Z"), Gate("H", ("a",))]
    z = (("a", "Z"),)
    twice = [*prep_h, Meas("m", z), Meas("m", z)]
    # A random outcome is postselected, then repeats must agree with it.
    assert not check_shot(twice, {"m": [1, 1]})
    assert check_shot(twice, {"m": [1, 0]})
    # A deterministic outcome must match, and a missing one is a problem.
    det = [Prep("a", "-Z"), Meas("m", z)]
    assert not check_shot(det, {"m": [1]})
    assert check_shot(det, {"m": [0]})
    assert check_shot(det, {})
    # A Pauli controlled by an earlier outcome corrects the state.
    cond = [
        Prep("a", "+X"),
        Meas("c", z),
        Cond("a", "X", ("c",)),
        Meas("m", z),
    ]
    assert not check_shot(cond, {"c": [1], "m": [0]})
    assert check_shot(cond, {"c": [1], "m": [1]})


RUNNERS = {
    "truth": run_truth,
    "tracking": run_tracking,
    "independence": run_independence,
    "ft": run_ft,
}


def _failing(case, decl):
    try:
        return set(RUNNERS[case.cls](case, decl))
    except Exception:  # noqa: BLE001 -- a codepack error is an outcome failure
        return {"outcome"}


@pytest.mark.parametrize("mutation_id", sorted(MUTATIONS))
def test_mutation_caught(mutation_id):
    mutation = MUTATIONS[mutation_id]
    decl = DECLARATIONS[mutation.codepack]
    cases = {c.id: c for c in generate(decl)}
    hook_errors: list[Exception] = []

    def guard(hook):
        # A hook that raises would otherwise read as a caught `outcome`.
        if hook is None:
            return None

        def run(*args):
            try:
                return hook(*args)
            except Exception as exc:
                hook_errors.append(exc)
                raise

        return run

    for case_id, targets in mutation.cases.items():
        case = cases[case_id]
        assert not set(targets) & set(declared_checks(decl, case_id))
        assert not set(targets) & _failing(case, decl)
        with engine.hooked(guard(mutation.code_hook), guard(mutation.op_hook)):
            failing = _failing(case, decl)
        assert not hook_errors
        assert set(targets) <= failing


def test_judge():
    judge(set(), set())
    judge({"O4"}, {"O4"})
    with pytest.raises(AssertionError):
        judge({"O4"}, {"O4", "O3"})  # a second bug hides behind a known one
    with pytest.raises(AssertionError):
        judge({"O4"}, set())  # a fix that flips nothing
