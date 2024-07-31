"""TODO"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from loqs.core import QuantumProgram
from loqs.core.instructions.instructionlabel import (
    InstructionLabelCastableTypes,
)

try:
    from pygsti.baseobjs import Label
    from pygsti.data import DataSet
    from pygsti.protocols import ExperimentDesign
    from pygsti.models import ExplicitOpModel
except ImportError as e:
    raise ImportError(
        "Could not import pygsti, needed for loqs.extenstions.pygstitools"
    ) from e


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
