"""TODO"""

from collections.abc import Mapping, Sequence
import functools
import itertools
import numpy as np
import numpy.typing as npt

from loqs.core import QuantumProgram
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)

try:
    import pygsti
    from pygsti.baseobjs import Label, ExplicitBasis
    from pygsti.data import DataSet
    from pygsti.protocols import ExperimentDesign
    from pygsti.models import ExplicitOpModel
except ImportError as e:
    raise ImportError(
        "Could not import pygsti, needed for loqs.extenstions.pygstitools"
    ) from e


## QUANTUMSIM BASIS TOOLS
def compute_qsim_bases(num_qubits: int):
    """TODO"""
    # Prep QuantumSim bases
    sig0q = np.array([[1.0, 0], [0, 0]], dtype="complex")
    sigXq = np.array([[0, 1], [1, 0]], dtype="complex") / np.sqrt(2)
    sigYq = np.array([[0, -1], [1, 0]], dtype="complex") * 1.0j / np.sqrt(2.0)
    sig1q = np.array([[0, 0], [0, 1]], dtype="complex")

    qbasis = itertools.product([sig0q, sigXq, sigYq, sig1q], repeat=num_qubits)
    qbasis = [functools.reduce(np.kron, x) for x in qbasis]

    return ExplicitBasis(
        qbasis,
        ["myEl%d" % i for i in range(4**num_qubits)],
        name=f"qsim_{num_qubits}q",
        longname=f"QuantumSim_{num_qubits}qubit",
    )


PYGSTI_QSIM_BASES = {nq: compute_qsim_bases(nq) for nq in [1, 2]}
"""Precomputed 1- and 2-qubit basis for QSim PTMs"""


def ptm_to_qsim_ptm(ptm: npt.NDArray):
    """TODO"""
    ptm = np.asarray(ptm)
    num_qubits = np.sqrt(np.log2(ptm.shape[0]))
    basis = pygsti.BuiltinBasis("pp", 4**num_qubits)
    return pygsti.tools.basistools.change_basis(
        ptm, basis, PYGSTI_QSIM_BASES[num_qubits]
    )


def unitary_to_qsim_ptm(U: npt.NDArray):
    """TODO"""
    ptm = np.asarray(pygsti.tools.unitary_to_pauligate(U))
    return ptm_to_qsim_ptm(ptm)


## EDESIGN CONVERSION TOOLS
def convert_edesign_to_programs(
    edesign: ExperimentDesign,
    model: ExplicitOpModel,
    physical_to_logical: Mapping[
        str | tuple, list[InstructionLabelCastableTypes]
    ],
    **kwargs,
) -> list[QuantumProgram]:
    """TODO"""
    label_to_logical = {Label(k): v for k, v in physical_to_logical.items()}

    if "name" in kwargs:
        del kwargs["name"]

    programs = []
    for circ in edesign.all_circuits_needing_data:
        completed_circ = model.complete_circuit(circ)

        stack = []
        for label in completed_circ._labels:
            stack.extend(label_to_logical[label])

        programs.append(QuantumProgram(stack, name=repr(circ), **kwargs))

    return programs


def convert_run_programs_to_dataset(
    edesign: ExperimentDesign, programs: Sequence[QuantumProgram]
) -> DataSet:
    """TODO"""
    from collections import Counter

    ds = DataSet()
    for circ, prog in zip(edesign.all_circuits_needing_data, programs):
        counts = Counter(prog.collect_shot_data("logical_measurement", -1))
        count_dict = {(str(k),): v for k, v in counts.items()}

        ds.add_count_dict(circ, count_dict)

    return ds
