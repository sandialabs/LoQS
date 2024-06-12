"""TODO
"""

from __future__ import annotations
from typing import Mapping, TypeAlias

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.instruction import InstructionParentTypes
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.recordables import MeasurementOutcomes


class QuantumLogicalOperation(Instruction):
    """TODO"""

    def __init__(
        self,
        physical_circuit: BasePhysicalCircuit,
        name: str = "(Unnamed quantum logical operation)",
        parent: InstructionParentTypes = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        self.physical_circuit = physical_circuit
        super().__init__(name, parent)

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {"state": BaseQuantumState}

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {"state": BaseQuantumState}

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """Map the input :class:`MockState` forward.

        This

        Parameters
        ----------
        input:
            The input frame/history information

        Returns
        -------
        output_frame:
            The new output frame
        """
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        state = last_frame["state"]

        # TODO: apply operations to new state
        # How do I know what kind of state?
        # How do I invalidate previous states if singleton?

        new_data = {
            "state": state,
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame

    def map_qubits(
        self, qubit_mapping: Mapping[str, str]
    ) -> QuantumLogicalOperation:
        mapped_circ = self.physical_circuit.map_qubit_labels(qubit_mapping)
        return QuantumLogicalOperation(mapped_circ, self.name, self.parent)


class QuantumClassicalLogicalOperation(QuantumLogicalOperation):
    """TODO"""

    def __init__(
        self,
        physical_circuit: BasePhysicalCircuit,
        name: str = "(Unnamed quantum-classical logical operation)",
        parent: InstructionParentTypes = None,
        reset_mcms: bool = True,
    ) -> None:
        super().__init__(physical_circuit, name, parent)
        self.reset_mcms = reset_mcms

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {
            "state": BaseQuantumState,
            "measurement_outcomes": MeasurementOutcomes,
        }

    def apply_unsafe(self, input: HistoryStackCastableTypes) -> HistoryFrame:
        """Map the input :class:`MockState` forward.

        This

        Parameters
        ----------
        input:
            The input frame/history information

        Returns
        -------
        output_frame:
            The new output frame
        """
        input = HistoryStack.cast(input)

        last_frame: HistoryFrame = input[-1]

        state = last_frame["state"]

        # TODO: apply operations to new state

        new_data = {
            "state": state,
            "measurement_outcomes": None,  # TODO
            "instruction": self,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame
