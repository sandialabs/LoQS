"""A LoQS QEC codepack for the 5-qubit code.

This implementation follows the construction in
Yoder, Takagi, and Chuang 2016 (arXiv:1603.03948).

TODO: Make this work with a general circuit backend
"""

import numpy as np

from loqs.backends.circuit import (
    BasePhysicalCircuit,
    PyGSTiPhysicalCircuit as PhysicalCircuit,
)
from loqs.core.qeccode import QECCode
from loqs.core.templatedcircuit import TemplatedCircuit


def create_qeccode(
    circuit_backend: type[BasePhysicalCircuit] = PhysicalCircuit,
):

    qec_operations = {}

    # Eqn 8 of 1603.03948
    stabilizers = ["ZZXIX", "XZZXI", "IXZZX", "XIXZZ"]
    templates = {k: _create_syndrome_check_circuit(k) for k in stabilizers}

    # In this case, we have one stage of stabilizer check per stabilizer
    # since we are doing ancilla reuse
    qubits = ["A0"] + [f"D{i}" for i in range(5)]
    checks = [{k: qubits} for k in stabilizers]

    # This circuit template will be each syndrome subsequently onto the single auxiliary qubit
    qec_operations["SE"] = TemplatedCircuit(
        templates,
        checks,
        qubit_labels=qubits,
        default_circuit_backend=circuit_backend,
    )

    # TODO: Logical state prep

    # TODO: Logical Z, X, and H

    # TODO: Logical CZ and CCZ

    return QECCode(qec_operations)


## Helper functions
def _create_syndrome_check_circuit(stabilizer: str):
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
