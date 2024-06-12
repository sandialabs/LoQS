"""A LoQS QEC codepack for the 5-qubit code.

TODO
"""

from collections.abc import Sequence
import itertools
import numpy as np

from loqs.backends.circuit import PyGSTiPhysicalCircuit as PhysicalCircuit
from loqs.core import Instruction, QECCode, TemplatedCircuit
from loqs.core.instructions import (
    CompositeInstruction,
    Decoder,
    PermutePatch,
    QuantumClassicalLogicalOperation,
    QuantumLogicalOperation,
    RepeatUntilSuccess,
    SyndromeExtraction,
)


def create_qec_code():
    """TODO"""

    qubits = ["A0", "A1"] + [f"D{i}" for i in range(5)]

    operations: dict[str, Instruction] = {}

    # Non-FT |-> state prep
    # First gray box of Fig 3 of arxiv:2208.01863
    nonft_state_prep_circ = PhysicalCircuit(
        [
            [
                ("Gh", "A0"),
                ("Gh", "D0"),
                ("Gh", "D1"),
                ("Gh", "D2"),
                ("Gh", "D3"),
                ("Gh", "D4"),
            ],
            [("Gcphase", "D0", "D1"), ("Gcphase", "D2", "D3")],
            [("Gcphase", "D1", "D2"), ("Gcphase", "D3", "D4")],
            [("Gcphase", "D0", "D4")],
        ],
        qubit_labels=qubits,
    )

    operations["Non-FT Minus Prep"] = QuantumLogicalOperation(
        nonft_state_prep_circ,
        name="Non-fault-tolerant |-> state prep",
    )

    # Try-until-success FT |-> state prep
    # First green box of Fig 3 of arxiv:2208.01863
    ft_state_prep_checks_circ = PhysicalCircuit(
        [
            # FT check 1
            ("Gcnot", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D4"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
            # FT check 2
            ("Gcphase", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcnot", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D2"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
            # FT check 3
            ("Gcphase", "A0", "D0"),
            ("Gcnot", "A0", "A1"),
            ("Gcnot", "A0", "D1"),
            ("Gcnot", "A0", "A1"),
            ("Gcphase", "A0", "D2"),
            ("Gh", "A0"),
            [("Iz", "A0"), ("Iz", "A1")],
        ],
        qubit_labels=qubits,
    )
    ft_state_prep_circ = nonft_state_prep_circ.append(
        ft_state_prep_checks_circ
    )
    ft_state_prep = QuantumClassicalLogicalOperation(
        ft_state_prep_circ,
        name="Non-fault-tolerant minus state prep",
        reset_mcms=True,
    )

    operations["FT Minus Prep"] = RepeatUntilSuccess(
        ft_state_prep,
        name="Repeat-until-success fault-tolerant minus state prep",
    )

    # Logical Z (transversal)
    # We use the Gottesman/standard convention here
    # Yoder convention only needed for CZ/CCZ
    logical_Z_circ = PhysicalCircuit(
        [[("Gzpi", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    operations["Z"] = QuantumLogicalOperation(logical_Z_circ, name="Logical Z")

    # Logical X (transversal)
    logical_X_circ = PhysicalCircuit(
        [[("Gxpi", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    operations["X"] = QuantumLogicalOperation(logical_X_circ, name="Logical X")

    # Logical H (transversal + permute)
    logical_H_circ = PhysicalCircuit(
        [[("Gxpi", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    logical_H_circ_inst = QuantumLogicalOperation(
        logical_H_circ, name="Logical H circuit"
    )
    # TODO: Double check this is the correct permutation in the Gottesman convention
    logical_H_permutation = PermutePatch(
        {
            "A0": "A1",
            "A1": "D2",
            "D1": "A1",
            "D2": "D1",
        },
        name="Logical H permutation",
    )

    operations["H"] = CompositeInstruction(
        [logical_H_circ_inst, logical_H_permutation], name="Logical H"
    )

    # Eqn 8 of arxiv:1603.03948
    stabilizers = ["ZZXIX", "XZZXI", "IXZZX", "XIXZZ"]
    syndrome_circuit = PhysicalCircuit([], qubit_labels=qubits)
    syndrome_qubits = {}
    for i, stab in enumerate(stabilizers):
        stabilizer_circuit = _create_syndrome_check_circuit(stab)
        syndrome_circuit.append_inplace(stabilizer_circuit)
        syndrome_qubits[stab] = ("A0", i)

    operations["SE"] = SyndromeExtraction(syndrome_circuit, syndrome_qubits)

    # Decoder (computed from commutation relations to stabilizers)
    lookup_table = _create_decoder_lookup(stabilizers)
    operations["Decode"] = Decoder(lookup_table)

    # TODO: Combined QEC instruction (and two round version)

    # TODO: Logical CZ and CCZ

    return QECCode(operations, qubits)


## Helper functions
def _create_syndrome_check_circuit(stabilizer: str) -> PhysicalCircuit:
    """TODO"""
    assert all(
        [p in "IXZ" for p in stabilizer]
    ), "Stabilizer must be Pauli string with only I, X, or Z"

    layers = []

    # We can do Z-type checks by CNOT from target to aux
    for Zloc in np.where([p == "Z" for p in stabilizer])[0]:
        layers.append(("Gcnot", f"D{Zloc}", "A0"))

    # We can go X-type checks by H on aux, CNOT from aux to target,
    # and then undoing the H
    layers.append(("Gh", "A0"))
    for Xloc in np.where([p == "X" for p in stabilizer])[0]:
        layers.append(("Gcnot", "A0", f"D{Xloc}"))
    layers.append(("Gh", "A0"))

    layers.append(("Iz", "A0"))

    qubits = ["A0"] + [f"D{i}" for i in range(len(stabilizer))]

    return PhysicalCircuit(layers, qubit_labels=qubits)


def _create_decoder_lookup(
    stabilizers: Sequence[str],
) -> dict[tuple[int], tuple[str, str]]:
    """TODO"""
    qubits = [f"D{i}" for i in range(len(stabilizers[0]))]
    reverse_lookup = {}
    for p, qi in itertools.product("XYZ", range(len(qubits))):
        syndrome = []
        for stab in stabilizers:
            if stab[qi] in (p, "I"):
                # Commute, should get 0
                syndrome.append(0)
            else:
                # Don't commute, should get 1
                syndrome.append(1)

        reverse_lookup[p, qubits[qi]] = tuple(syndrome)

    lookup = {v: k for k, v in reverse_lookup.items()}
    return lookup
