"""A LoQS QEC codepack for the 5-qubit code.

This implementation follows the construction in
Yoder, Takagi, and Chuang 2016 (arXiv:1603.03948).

TODO: Make this work with a general circuit backend
"""

from collections.abc import Sequence
import itertools
import numpy as np

from loqs.backends.circuit import (
    BasePhysicalCircuit,
    PyGSTiPhysicalCircuit as PhysicalCircuit,
)
from loqs.core import Instruction, QECCode, TemplatedCircuit
from loqs.core.instructions import Decoder, SyndromeExtraction


def create_qeccode(
    circuit_backend: type[BasePhysicalCircuit] = PhysicalCircuit,
):
    # TODO: Make this work for more than just the pygsti backend

    instructions: dict[str, Instruction] = {}

    # Eqn 8 of 1603.03948
    stabilizers = ["ZZXIX", "XZZXI", "IXZZX", "XIXZZ"]
    templates = {k: _create_syndrome_check_circuit(k) for k in stabilizers}

    # In this case, we have one stage of stabilizer check per stabilizer
    # since we are doing ancilla reuse
    qubits = ["A0"] + [f"D{i}" for i in range(5)]
    checks = [{k: qubits} for k in stabilizers]

    # This circuit template will be each syndrome subsequently onto the single auxiliary qubit
    se_circuit = TemplatedCircuit(
        templates,
        checks,
        qubit_labels=qubits,
        default_circuit_backend=circuit_backend,
    )
    syndrome = {k: "A0" for k in stabilizers}

    instructions["SE"] = SyndromeExtraction(se_circuit, syndrome)

    # Decoder (computed from commutation relations to stabilizers)
    lookup_table = _create_decoder_lookup(stabilizers)
    instructions["Decode"] = Decoder(lookup_table)

    # Combined QEC instruction
    # TODO Composite instruction

    # TODO: Logical state prep

    # TODO: Logical Z, X, and H

    # TODO: Logical CZ and CCZ

    return QECCode(instructions)


## Helper functions
def _create_syndrome_check_circuit(stabilizer: str) -> PhysicalCircuit:
    """TODO"""
    assert all(
        [p in "IXZ" for p in stabilizer]
    ), "Stabilizer must be Pauli string with only I, X, or Z"

    layers = []

    # We can do Z-type checks by CNOT from target to aux
    for Zloc in np.where([p == "Z" for p in stabilizer])[0]:
        layers.append(("Gcnot", f"Q{Zloc}", "Qaux"))

    # We can go X-type checks by H on aux, CNOT from aux to target,
    # and then undoing the H
    layers.append(("Gh", "Qaux"))
    for Xloc in np.where([p == "X" for p in stabilizer])[0]:
        layers.append(("Gcnot", "Qaux", f"Q{Xloc}"))
    layers.append(("Gh", "Qaux"))

    qubits = ["Qaux"] + [f"Q{i}" for i in range(len(stabilizer))]

    return PhysicalCircuit(layers, qubit_labels=qubits)


def _create_decoder_lookup(
    stabilizers: Sequence[str],
) -> dict[tuple[int], tuple[str, str]]:
    """TODO"""
    qubits = [f"Q{i}" for i in range(len(stabilizers[0]))]
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
