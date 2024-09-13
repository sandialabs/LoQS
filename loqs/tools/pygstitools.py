"""TODO"""

from collections.abc import Mapping, Sequence
import numpy as np
import numpy.typing as npt

from loqs.backends.model.pygstimodel import PYGSTI_QSIM_BASES
from loqs.core import QuantumProgram
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)

try:
    import pygsti
    from pygsti.baseobjs import Label, ExplicitBasis
    from pygsti.circuits import Circuit
    from pygsti.data import DataSet
    from pygsti.protocols import ExperimentDesign
    from pygsti.models import ExplicitOpModel
except ImportError as e:
    raise ImportError(
        "Could not import pygsti, needed for loqs.extenstions.pygstitools"
    ) from e


## QUANTUMSIM BASIS TOOLS
def ptm_to_qsim_ptm(ptm: npt.NDArray):
    """TODO"""
    ptm = np.asarray(ptm)
    num_qubits = np.log2(ptm.shape[0]) // 2
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
        completed_circ: Circuit = model.complete_circuit(circ)

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
