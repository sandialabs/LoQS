"""Tests specific to loqs.backends.circuit.stimcircuit.

Behaviour shared with the other circuit backends is in test_circuits.py.
"""

import pytest

stim = pytest.importorskip("stim")

from loqs.backends.circuit.stimcircuit import (  # noqa: E402
    STIMPhysicalCircuit,
    _get_used_stim_indices,
    _reindex_stim_circuit,
)


def test_init():
    sparse_circuit_str = "H 0\nTICK\nCX 0 5"
    custom_labels = ["Q0", "Q5"]

    # Sparse indices become compact, with the used indices as labels
    circ1 = STIMPhysicalCircuit(sparse_circuit_str)
    assert circ1.qubit_labels == [0, 5]
    assert circ1.circuit.num_qubits == 2

    # Explicit labels, for a string and for a stim.Circuit
    circ2 = STIMPhysicalCircuit("H 0\nTICK\nCX 0 1", custom_labels)
    assert circ2.qubit_labels == custom_labels
    assert circ2.circuit.num_qubits == 2

    stim_circ = stim.Circuit(sparse_circuit_str)
    circ3 = STIMPhysicalCircuit(stim_circ)
    assert circ3.qubit_labels == [0, 5]
    assert circ3.circuit.num_qubits == 2

    circ4 = STIMPhysicalCircuit(stim_circ, custom_labels)
    assert circ4.qubit_labels == custom_labels
    assert circ4.circuit.num_qubits == 2

    # Copy, with and without new labels
    circ5 = STIMPhysicalCircuit(circ1)
    assert circ5.qubit_labels == circ1.qubit_labels
    assert str(circ5.circuit) == str(circ1.circuit)

    circ6 = STIMPhysicalCircuit(circ1, ["A", "B"])
    assert circ6.qubit_labels == ["A", "B"]
    assert circ6.circuit.num_qubits == 2

    # copy() keeps the circuit as it is, for any integer labels
    circ7 = STIMPhysicalCircuit("H 0\nTICK\nCX 0 1", [1, 0])
    assert circ7.copy().circuit == circ7.circuit
    assert circ7.copy().qubit_labels == [1, 0]


@pytest.mark.parametrize("case", ["fewer", "past-end"])
def test_wrong_label_count_raises(case):
    if case == "fewer":
        circ = STIMPhysicalCircuit("H 0\nTICK\nCX 0 5")
        with pytest.raises(ValueError):
            STIMPhysicalCircuit(circ, ["A"])
        with pytest.raises(ValueError):
            circ.set_qubit_labels_inplace(["A"])
    else:
        with pytest.raises(ValueError):
            STIMPhysicalCircuit(stim.Circuit("CX 0 5\nTICK"), ["A", "B", "C"])


def test_unused_labels():
    circ = STIMPhysicalCircuit(stim.Circuit("H 0\nTICK"), [0, 1])
    assert circ.qubit_labels == [0, 1]
    assert circ.circuit == stim.Circuit("H 0\nTICK")

    # More labels than used indices is positional: index 1 acts on "B"
    circ = STIMPhysicalCircuit(stim.Circuit("H 1\nTICK"), ["A", "B"])
    assert circ.circuit == stim.Circuit("H 1\nTICK")
    copied = circ.copy()
    assert copied.qubit_labels == ["A", "B"]
    assert copied.circuit == circ.circuit


def test_sparse_circuit_compactness():
    sparse_circuit_str = "H 0\nTICK\nCX 0 10\nTICK\nM 10"
    circ = STIMPhysicalCircuit(sparse_circuit_str)

    assert circ.qubit_labels == [0, 10]
    assert circ.circuit.num_qubits == 2

    # The internal circuit uses the compact indices 0 and 1
    circ_str = str(circ.circuit)
    assert "H 0" in circ_str
    assert "CX 0 1" in circ_str
    assert "M 1" in circ_str
    assert "H 10" not in circ_str

    circ.delete_qubits_inplace([10])
    assert circ.circuit.num_qubits == 1
    assert circ.qubit_labels == [0]

    other_circ = STIMPhysicalCircuit("X 10\nTICK\nY 15")
    circ.merge_inplace(other_circ, 0)
    assert circ.circuit.num_qubits == 3
    assert set(circ.qubit_labels) == {0, 10, 15}


def test_string_label_substitution():
    # Labels in measurements and in instructions with arguments map to
    # STIM indices
    circ = STIMPhysicalCircuit(
        "H Q0\nTICK\nM Q0\nTICK\nMR Q1\nTICK\nX_ERROR(0.1) Q0", ["Q0", "Q1"]
    )
    circ_str = str(circ.circuit)
    assert "H 0" in circ_str
    assert "M 0" in circ_str
    assert "MR 1" in circ_str
    assert "X_ERROR(0.1) 0" in circ_str

    # The CNOT alias is substituted
    aliased = STIMPhysicalCircuit.substitute_command_aliases(
        "CNOT 0 1\nTICK\nH 0"
    )
    assert "CX 0 1" in aliased
    assert "CNOT" not in aliased


@pytest.mark.parametrize(
    "circ_str, counts",
    [
        pytest.param(
            "REPEAT 2 {\n  H 0\n  TICK\n}\n"
            "X 0\nTICK\n"
            "REPEAT 3 {\n  Y 0\n  TICK\n}",
            {"H 0": 2, "Y 0": 3, "X 0": 1},
            id="sibling",
        ),
        pytest.param(
            "REPEAT 2 {\n  REPEAT 3 {\n    H 0\n    TICK\n  }\n}",
            {"H 0": 6},
            id="nested",
        ),
    ],
)
def test_unroll_repeats(circ_str, counts):
    unrolled = STIMPhysicalCircuit(circ_str, ["Q0"])._unroll_repeats()
    assert "REPEAT" not in unrolled
    for gate, count in counts.items():
        assert unrolled.count(gate) == count


def test_no_tick_warning_emitted():
    with pytest.warns(UserWarning):
        STIMPhysicalCircuit("H 0\nCX 0 1", ["Q0", "Q1"])


@pytest.mark.parametrize("case", ["mpp", "merge-collision"])
def test_unsupported_input_raises(case):
    if case == "mpp":
        with pytest.raises(ValueError):
            STIMPhysicalCircuit("MPP 0 1\nTICK")
    else:
        circ1 = STIMPhysicalCircuit("H 0\nTICK\nX 0", ["Q0"])
        circ2 = STIMPhysicalCircuit("Y 0\nTICK\nZ 0", ["Q0"])
        with pytest.raises(ValueError):
            circ1.merge_inplace(circ2, 0)


def test_layer_edits_unroll_repeats_and_tagged_ticks():
    circ = STIMPhysicalCircuit(
        "H 0\nTICK[a]\nREPEAT 2 {\n    X 0\n    TICK\n}\nY 1",
        ["Q0", "Q1"],
    )
    assert circ.get_possible_discrete_error_locations() == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 1),
    ]

    base = STIMPhysicalCircuit("Z 1\nTICK\nZ 1\nTICK", ["Q0", "Q1"])
    assert base.insert(circ, 1).circuit == stim.Circuit(
        "Z 1\nTICK\nH 0\nTICK[a]\nX 0\nTICK\nX 0\nTICK\nY 1\nTICK\nZ 1\nTICK"
    )

    # Open-ended target; the reference's X layers are its idles
    target = STIMPhysicalCircuit("H 0\nTICK\nTICK\nTICK\nY 1", ["Q0", "Q1"])
    target.transplant_idle_schedule_inplace(circ, ["Q0"], ["X"])
    assert target.circuit == stim.Circuit(
        "H 0\nTICK\nX 0\nTICK\nX 0\nTICK\nY 1"
    )

    circ.pad_single_qubit_idles_by_duration_inplace(
        {1: "I"}, {}, default_duration=1
    )
    assert circ.circuit == stim.Circuit(
        "H 0\nI 1\nTICK[a]\nX 0\nI 1\nTICK\nX 0\nI 1\nTICK\nY 1\nI 0"
    )


def test_get_possible_discrete_error_locations_inverted_target():
    # An inverted target flips the recorded outcome, not the measured
    # qubit, so it resolves like plain "M 0"
    circ = STIMPhysicalCircuit("M !0\nTICK", ["Q0"])

    locations = circ.get_possible_discrete_error_locations()

    assert len(locations) == 1
    _, qubit_info = locations[0]
    assert qubit_info == 0
    assert isinstance(qubit_info, int)


def test_helper_functions():
    circ = stim.Circuit("H 0\nCX 0 5\nM 3")
    assert _get_used_stim_indices(circ) == [0, 3, 5]

    reindexed = _reindex_stim_circuit(circ, {0: 0, 3: 1, 5: 2})
    circ_str = str(reindexed)
    assert "H 0" in circ_str
    assert "CX 0 2" in circ_str
    assert "M 1" in circ_str
    assert reindexed.num_qubits == 3

    # An inverted target is kept
    circ_inv = stim.Circuit("H 0\nM !1\nCX 0 2")
    assert _get_used_stim_indices(circ_inv) == [0, 1, 2]
    reindexed_inv = _reindex_stim_circuit(circ_inv, {0: 0, 1: 1, 2: 2})
    assert "M !1" in str(reindexed_inv)

    # Record targets are not qubits
    circ_rec = stim.Circuit("H 0\nM 0\nDETECTOR rec[-1]")
    assert _get_used_stim_indices(circ_rec) == [0]

    # Gate arguments stay attached to the instruction name
    circ_args = stim.Circuit(
        "X_ERROR(0.25) 0\nPAULI_CHANNEL_1(0.1, 0.2, 0.3) 1"
    )
    reindexed_args = _reindex_stim_circuit(circ_args, {0: 1, 1: 0})
    args_str = str(reindexed_args)
    assert "X_ERROR(0.25) 1" in args_str
    assert "PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0" in args_str


def test_repeat_block_indices_and_reindexing():
    circ = stim.Circuit("X 0\nREPEAT 3 {\nCX 0 1\n}\nM 1")
    assert _get_used_stim_indices(circ) == [0, 1]

    reindexed = _reindex_stim_circuit(circ, {0: 1, 1: 0})
    circ_str = str(reindexed)
    assert "X 1" in circ_str
    assert "REPEAT 3 {" in circ_str
    assert "CX 1 0" in circ_str
    assert "M 0" in circ_str
    assert reindexed[1].repeat_count == 3
