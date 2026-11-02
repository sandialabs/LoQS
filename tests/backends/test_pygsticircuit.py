"""Tester for loqs.backends.circuit.pygsticircuit.PyGSTiPhysicalCircuit"""

import pytest

pygsti = pytest.importorskip("pygsti")
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit

from loqs.backends import (
    ListPhysicalCircuit,
    PyGSTiPhysicalCircuit as PhysCirc,
)


def _check(circ, expected_circ):
    assert circ.circuit == expected_circ
    assert list(circ.qubit_labels) == list(expected_circ.line_labels)


def test_init():
    # Every layer form in LayerTypes
    test_circ = Circuit(
        [
            "Gidle",
            ("Gxpi2", "Q0"),
            ("Gypi2", "Q1"),
            ("Gcnot", "Q0", "Q1"),
            [("Gxpi2", "Q0"), ("Gypi2", "Q1")],
            Label("Gxpi2", ("Q0",)),
        ],
        line_labels=["Q0", "Q1"],
    )  # type: ignore

    _check(PhysCirc(test_circ, test_circ.line_labels), test_circ)

    # Implicit qubit labels
    _check(PhysCirc(test_circ), test_circ)

    # String and layer-tuple forms
    _check(PhysCirc(test_circ.str), test_circ)
    _check(PhysCirc(test_circ.layertup), test_circ)


def test_init_allows_nonconforming_qubit_labels():
    # pyGSTi's own line-label restriction only allows labels starting with
    # one of Q/T/L/A/D; seam-qubit ("S0") and multi-letter ("AUX0") labels
    # must survive construction, mapping and copying.
    pc = PhysCirc([("Gcnot", "AUX0", "S0")])
    expected = Circuit(
        [("Gcnot", "AUX0", "S0")], line_labels=["AUX0", "S0"], check=False
    )  # type: ignore
    _check(pc, expected)

    pc.map_qubit_labels_inplace({"S0": "S1"})
    assert list(pc.qubit_labels) == ["AUX0", "S1"]
    assert list(pc.copy().qubit_labels) == ["AUX0", "S1"]

    lc = ListPhysicalCircuit([[("Gxpi2", ("S0",))]], ["S0"])
    expected = Circuit(
        [("Gxpi2", "S0")], line_labels=["S0"], check=False
    )  # type: ignore
    _check(PhysCirc(lc), expected)


def test_serialization_gatename_needing_pygsti_safe_alias(make_temp_path):
    # "GiMCM" doesn't survive pyGSTi's own string-form circuit parser
    # unaliased -- it truncates a gate name at its first uppercase
    # letter after the initial character, so "GiMCM" would otherwise
    # round-trip as "Gi" and collide with the real "Gi" gate below.
    circ = Circuit(
        [[("GiMCM", "Q0")], [("Gi", "Q0")]], line_labels=["Q0"]
    )  # type: ignore
    pc = PhysCirc(circ)

    serialized = pc._serialize_circuit()
    assert isinstance(serialized, dict)
    assert serialized["gatename_renames"] == {"GiMCM": "Gimcm"}

    with make_temp_path(suffix=".json") as tmp_path:
        pc.write(tmp_path)
        pc2 = PhysCirc.read(tmp_path)

    _check(pc2, circ)


def test_round_trip_equal_with_unsorted_parallel_layer(make_temp_path):
    circ = PhysCirc([[("Gxpi2", "Q1")], [("Gcnot", "Q0", "Q1")]], ["Q0", "Q1"])
    circ.merge_inplace(PhysCirc([("Gypi2", "Q0")], ["Q0", "Q1"]), 0)

    with make_temp_path(suffix=".json") as tmp_path:
        circ.write(tmp_path)
        loaded = PhysCirc.read(tmp_path)

    assert loaded == circ
    assert hash(loaded) == hash(circ)


def test_serialization_gatename_alias_collision_raises():
    # "GiMCM" and "Gimcm" both alias to the same pyGSTi-safe name --
    # serializing both in one circuit can't be reversed unambiguously.
    circ = Circuit(
        [[("GiMCM", "Q0")], [("Gimcm", "Q0")]], line_labels=["Q0"]
    )  # type: ignore
    pc = PhysCirc(circ)
    with pytest.raises(ValueError):
        pc._serialize_circuit()
