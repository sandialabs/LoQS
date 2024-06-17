"""A LoQS QEC codepack for the 5-qubit code.

TODO
"""

from collections.abc import Sequence
import itertools
import numpy as np

from loqs.backends.circuit import PyGSTiPhysicalCircuit as PhysicalCircuit
from loqs.core import Instruction, QECCode
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

    # TODO: Verify this is actually minus
    operations["Non-FT Minus Prep"] = QuantumLogicalOperation(
        nonft_state_prep_circ,
        name="Non-fault-tolerant minus state prep",
        fault_tolerant=False,
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
        fault_tolerant=True,
        reset_mcms=True,
    )

    operations["FT Minus Prep"] = RepeatUntilSuccess(
        ft_state_prep,
        name="Repeat-until-success fault-tolerant minus state prep",
    )

    # Logical X (transversal)
    # Eqn B1 of arxiv:2208.01863
    logical_X_circ = PhysicalCircuit(
        [[("Gypi", "D0"), ("Gxpi", "D2"), ("Gypi", "D4")]], qubit_labels=qubits
    )
    operations["X"] = QuantumLogicalOperation(
        logical_X_circ, name="Logical X", fault_tolerant=True
    )

    # Logical Z (transversal)
    # Eqn B3 of arxiv:2208.01863
    logical_Z_circ = PhysicalCircuit(
        [[("Gxpi", "D0"), ("Gzpi", "D2"), ("Gxpi", "D4")]], qubit_labels=qubits
    )
    operations["Z"] = QuantumLogicalOperation(
        logical_Z_circ, name="Logical Z", fault_tolerant=True
    )

    # Logical H (transversal + permute)
    # Fig 2b of arxiv:1603.03948
    logical_H_circ = PhysicalCircuit(
        [[("Gh", q) for q in qubits[2:]]], qubit_labels=qubits
    )
    logical_H_circ_inst = QuantumLogicalOperation(
        logical_H_circ, name="Logical H circuit", fault_tolerant=True
    )
    logical_H_permutation = PermutePatch(
        {  # final: initial
            "D0": "D1",
            "D1": "D4",
            # D2 is unpermuted
            "D3": "D0",
            "D4": "D3",
        },
        name="Logical H permutation",
    )

    operations["H"] = CompositeInstruction(
        [logical_H_circ_inst, logical_H_permutation], name="Logical H"
    )

    # Eqn B4-B7 of arxiv:2208.01863
    stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    syndrome_circuit = PhysicalCircuit([], qubit_labels=qubits)
    syndrome_qubits = {}
    for i, stab in enumerate(stabilizers):
        stabilizer_circuit = _create_syndrome_check_circuit(stab)
        syndrome_circuit.append_inplace(stabilizer_circuit)
        syndrome_qubits[stab] = ("A0", i)

    # TODO: Current SE is not FT
    operations["SE"] = SyndromeExtraction(
        syndrome_circuit, syndrome_qubits, fault_tolerant=False
    )

    # Decoder (computed from commutation relations to stabilizers)
    lookup_table = _create_decoder_lookup(stabilizers)
    operations["Decode"] = Decoder(lookup_table)

    # TODO: Combined QEC instruction (and two round version)

    # TODO: Logical CZ and CCZ

    return QECCode(operations, qubits)


## Helper functions
def _create_syndrome_check_circuit(stabilizer: str) -> PhysicalCircuit:
    """TODO

    Essentially the Wikipedia version of syndrome extraction.
    TODO: Not FT to X errors on auxiliary?
    Either use 2 qubits or use Las Vegas approach?
    """
    assert all(
        [p in "IXZ" for p in stabilizer]
    ), "Stabilizer must be Pauli string with only I, X, or Z"

    layers = []

    layers.append(("Gh", "A0"))

    for Zloc in np.where([p == "Z" for p in stabilizer])[0]:
        layers.append(("Gcphase", "A0", f"D{Zloc}"))

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
