"""Tester for loqs.codepacks.generator.generate"""

import dataclasses

import pytest

stim = pytest.importorskip("stim")

from loqs.codepacks.generator import gates  # noqa: E402
from loqs.codepacks.generator.derive import tableau  # noqa: E402
from loqs.codepacks.generator.generate import generate  # noqa: E402
from loqs.codepacks.generator.schema import Circuit, SpecError  # noqa: E402
from loqs.codepacks.steane.spec import SPEC  # noqa: E402


def _with_item(spec, key, **changes):
    items = tuple(
        dataclasses.replace(it, **changes) if it.key == key else it
        for it in spec.items
    )
    return dataclasses.replace(spec, items=items)


def _two_logicals():
    facts = dataclasses.replace(
        SPEC.facts,
        logicals=SPEC.facts.logicals + (("IIIIXXX", "IIIIZZZ"),),
    )
    return dataclasses.replace(SPEC, facts=facts)


WRONG_SPECS = {
    "s-as-s-dag": (_with_item(SPEC, "S", logical="S_DAG"), SpecError),
    "not-code-preserving": (
        _with_item(SPEC, "Sdag", text="Gzpi2 D0"),
        SpecError,
    ),
    "unknown-insert": (
        _with_item(
            SPEC, "Flagged S2-S3-S4 Feed-Forward", inserts=("QEC decoder",)
        ),
        SpecError,
    ),
    "unknown-gate": (
        _with_item(SPEC, "Raw Z Data Measure", text="Gfoo D0"),
        SpecError,
    ),
    "duplicate-key": (
        dataclasses.replace(
            SPEC,
            items=SPEC.items
            + (Circuit(SPEC.items[0].key, "Gxpi D0", SPEC.items[0].src),),
        ),
        SpecError,
    ),
    "shared-update-key-differs": (
        _with_item(SPEC, "H", update_key="S Frame Update"),
        SpecError,
    ),
    "two-logicals": (_two_logicals(), NotImplementedError),
    "cnot-in-gate": (
        _with_item(SPEC, "I", text="Gcnot D0 D1\nGcnot D0 D1"),
        NotImplementedError,
    ),
}


@pytest.mark.parametrize("wrong", WRONG_SPECS.values(), ids=WRONG_SPECS)
def test_wrong_spec_refuses(wrong):
    spec, error = wrong
    with pytest.raises(error):
        generate(spec)


def test_gate_table_matches_stim():
    for name, entry in gates.GATES.items():
        if entry.kind == "instrument":
            continue
        assert tableau(name) == stim.Tableau.from_named_gate(entry.unitary)
