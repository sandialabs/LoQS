"""TODO
"""

from __future__ import annotations
from typing import Mapping, TypeAlias

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model.basemodel import (
    BaseNoiseModel,
    InstrumentRep,
    GateRep,
)
from loqs.backends.state import BaseQuantumState
from loqs.core import Instruction, HistoryStack, HistoryFrame
from loqs.core.instruction import InstructionParentTypes
from loqs.core.history import HistoryStackCastableTypes
from loqs.core.recordables import MeasurementOutcomes


LogicalOperationCastableTypes: TypeAlias = (
    "QuantumLogicalOperation | BasePhysicalCircuit"
)


class QuantumLogicalOperation(Instruction):
    """TODO"""

    def __init__(
        self,
        physical_circuit: LogicalOperationCastableTypes,
        name: str = "(Unnamed quantum logical operation)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
    ) -> None:
        """TODO

        Parameters
        ----------
        """
        super().__init__(name, parent, fault_tolerant)

        if isinstance(physical_circuit, QuantumLogicalOperation):
            self.physical_circuit = physical_circuit.physical_circuit
        elif isinstance(physical_circuit, BasePhysicalCircuit):
            self.physical_circuit = physical_circuit
        else:
            raise ValueError(
                f"Cannot create QuantumLogicalOperation from {physical_circuit}"
            )

    @property
    def input_frame_spec(self) -> dict[str, type]:
        return {"state": BaseQuantumState}

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {"state": BaseQuantumState}

    def apply_unsafe(
        self,
        input: HistoryStackCastableTypes,
        model: BaseNoiseModel,
        inplace: bool = True,
    ) -> HistoryFrame:
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
        # Check model can take our circuit
        assert (
            type(self.physical_circuit) in model.input_circuit_types
        ), "Physical circuit type not allowed as model input"

        input = HistoryStack.cast(input)
        last_frame: HistoryFrame = input[-1]

        # This will only work if state is already a BaseQuantumState-derived object
        # But that should basically always be the case
        state = BaseQuantumState.cast(last_frame["state"])

        new_state, _ = self._propogate_state(state, model, inplace)

        new_data = {
            "state": new_state,
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

    def _propogate_state(
        self, state: BaseQuantumState, model: BaseNoiseModel, inplace: bool
    ) -> tuple[BaseQuantumState, MeasurementOutcomes]:
        # Find a compatible model/state oprep
        oprep: GateRep | None = None
        for rep in model.output_gate_reps:
            if rep in state.input_reps:
                oprep = rep
        assert (
            oprep is not None
        ), "Could not find matching gate rep between model output and state input"

        instrep: InstrumentRep | None = None
        for rep in model.output_instrument_reps:
            if rep in state.input_reps:
                instrep = rep
        assert (
            instrep is not None
        ), "Could not find matching instrument rep between model output and state input"

        # Look up reps from model
        reps = model.get_reps(self.physical_circuit, oprep, instrep)

        # Apply operator reps to state
        if inplace:
            outcomes = state.apply_reps_inplace(reps)
        else:
            state, outcomes = state.apply_reps(reps)

        return state, MeasurementOutcomes(outcomes)


class QuantumClassicalLogicalOperation(QuantumLogicalOperation):
    """TODO"""

    def __init__(
        self,
        physical_circuit: LogicalOperationCastableTypes,
        name: str = "(Unnamed quantum-classical logical operation)",
        parent: InstructionParentTypes = None,
        fault_tolerant: bool | None = None,
        reset_mcms: bool = True,
    ) -> None:
        super().__init__(physical_circuit, name, parent, fault_tolerant)

        self.reset_mcms = reset_mcms

    @property
    def output_frame_spec(self) -> dict[str, type]:
        return {
            "state": BaseQuantumState,
            "measurement_outcomes": MeasurementOutcomes,
        }

    def apply_unsafe(
        self,
        input: HistoryStackCastableTypes,
        model: BaseNoiseModel,
        inplace: bool = True,
    ) -> HistoryFrame:
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
        # Check model can take our circuit
        assert (
            type(self.physical_circuit) in model.input_circuit_types
        ), "Physical circuit type not allowed as model input"

        input = HistoryStack.cast(input)
        last_frame: HistoryFrame = input[-1]

        # This will only work if state is already a BaseQuantumState-derived object
        # But that should basically always be the case
        state = BaseQuantumState.cast(last_frame["state"])

        new_state, outcomes = self._propogate_state(state, model, inplace)

        new_data = {
            "state": new_state,
            "instruction": self,
            "measurement_outcomes": outcomes,
        }

        output_frame = last_frame.update(
            new_data=new_data, new_log=f"{self.name} result"
        )
        return output_frame

    def map_qubits(
        self, qubit_mapping: Mapping[str, str]
    ) -> QuantumClassicalLogicalOperation:
        mapped_circ = self.physical_circuit.map_qubit_labels(qubit_mapping)
        return QuantumClassicalLogicalOperation(
            mapped_circ, self.name, self.parent, self.reset_mcms
        )
