"""Tester for loqs.backends.circuit.listcircuit.ListPhysicalCircuit"""

from unittest import mock

import pytest

from loqs.backends import ListPhysicalCircuit as PhysCirc


def _check(circ, expected_circ, expected_labels):
    for l1, l2 in zip(circ.circuit, expected_circ):
        assert set(l1) == set(l2)
    assert set(circ.qubit_labels) == set(expected_labels)


def test_init():
    # Every layer form in LayerTypes
    test_circ = [
        ("Gxpi2", "Q0"),
        ("Gypi2", "Q1"),
        ("Gcnot", ["Q0", "Q1"]),
        [("Gxpi2", "Q0"), ("Gypi2", "Q1")],
    ]
    expected = [
        [("Gxpi2", ("Q0",))],
        [("Gypi2", ("Q1",))],
        [("Gcnot", ("Q0", "Q1"))],
        [("Gxpi2", ("Q0",)), ("Gypi2", ("Q1",))],
    ]
    labels = ("Q0", "Q1")

    _check(PhysCirc(test_circ, labels), expected, labels)

    # Implicit qubit labels
    _check(PhysCirc(test_circ), expected, labels)


def test_init_from_pygsti_circuit_does_not_reconstruct_it():
    """An existing `PyGSTiPhysicalCircuit` is read directly, without
    constructing or copying a second one."""
    pytest.importorskip("pygsti")
    from loqs.backends import PyGSTiPhysicalCircuit

    pgc = PyGSTiPhysicalCircuit([("Gxpi2", "Q0")], ["Q0"])
    original_init = PyGSTiPhysicalCircuit.__init__
    call_count = [0]

    def counting_init(self, *args, **kwargs):
        call_count[0] += 1
        return original_init(self, *args, **kwargs)

    with mock.patch.object(PyGSTiPhysicalCircuit, "__init__", counting_init):
        PhysCirc(pgc)
    assert call_count[0] == 0


def test_tiling_padding_relabelling_agree_with_pygsti():
    """Tiling a template with a two-qubit gate, staggered tiles, idle
    padding by duration and relabelling gives the same layers whether
    built in PyGSTi or in List circuits."""
    pytest.importorskip("pygsti")
    from loqs.backends import PyGSTiPhysicalCircuit

    layers = [
        [("Gh", ("A",)), ("Gx", ("B",))],
        [("Gcnot", ("A", "B"))],
        [("Gx", ("A",))],
    ]
    labels = ["Q0", "Q1", "Q2", "Q3"]
    tile_qubits = [["Q0", "Q1"], ["Q2", "Q3"]]
    relabel = {"Q0": "D0", "Q1": "D1", "Q2": "A0", "Q3": "A1"}

    def build(circ_type, template):
        tiled = circ_type.from_circuit_tiling(
            template, labels, tile_qubits, merge_offsets=[0, 1]
        )
        padded = tiled.pad_single_qubit_idles_by_duration(
            {1: "Gi1", 2: "Gi2"}, {"Gh": 1, "Gx": 2, "Gcnot": 2}
        )
        return padded.map_qubit_labels(relabel)

    list_template = PhysCirc(layers, ["A", "B"])
    list_built = build(PhysCirc, list_template)
    pygsti_built = build(
        PyGSTiPhysicalCircuit, PyGSTiPhysicalCircuit(list_template)
    )

    from_pygsti = PhysCirc(pygsti_built)
    assert from_pygsti.depth == list_built.depth == 4
    assert set(from_pygsti.qubit_labels) == set(list_built.qubit_labels)
    for layer_a, layer_b in zip(from_pygsti.circuit, list_built.circuit):
        assert set(layer_a) == set(layer_b)

    # Guard against a vacuous comparison: the interesting labels exist
    flat = {label for layer in list_built.circuit for label in layer}
    assert ("Gcnot", ("A0", "A1")) in flat
    assert ("Gi2", ("A0",)) in flat
