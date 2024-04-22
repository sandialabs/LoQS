""":class:`QSimQuantumState` definition.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Type, TypeAlias, Union

from loqs.backends.model import OpRep
from loqs.backends.state import BaseQuantumState


class QSimQuantumState(BaseQuantumState):
    """Base class for an object that holds a QuantumSim SparseDM state."""

    @property
    def name(self) -> str:
        return "QuantumSim"

    @property
    def QubitTypes(self) -> TypeAlias:
        # Technically not a true restriction of SparseDM, but keeping it simple
        return Union[str, int]

    @property
    def OpRepInputs(self) -> Iterable[OpRep]:
        return [OpRep.PTM_QSIM]

    @property
    def StateType(self) -> Type:
        try:
            from quantumsim.sparsedm import SparseDM
        except ImportError as e:
            raise ImportError("Failed import, cannot use QuantumSim as backend") from e

        return SparseDM

    def apply_operator_reps_inplace(self, op_reps: Iterable) -> None:
        # TODO: Instruments/measurements
        for rep, qubits in op_reps:
            if len(qubits) == 1:
                self.state.apply_ptm(qubits[0], rep)
            elif len(qubits) == 2:
                self.state.apply_two_ptm(qubits[0], qubits[1], rep)
            else:
                raise ValueError("Cannot apply more than a 2 qubit operation")

    def apply_operator_reps(self, op_reps: Iterable) -> QSimQuantumState:
        return super().apply_operator_reps(op_reps)

    def copy(self) -> QSimQuantumState:
        return QSimQuantumState(self.state.copy())

    def __init__(
        self,
        state: Union[StateType, QSimQuantumState, int],
        qubit_labels: Optional[Iterable[QubitTypes]] = None,
    ) -> None:
        """Initialize a BaseQuantumState.

        Parameters
        ----------
        state:
            A representation of the underlying state. If an integer is passed,
            an all-0 state with that number of qubits is passed

        qubit_labels:
            Optional qubit labels. If not provided, the default range of ints
            is used.
        """
        try:
            from quantumsim.sparsedm import SparseDM
        except ImportError as e:
            raise ImportError("Failed import, cannot use QuantumSim as backend") from e
        
        if isinstance(state, QSimQuantumState):
            self.state = state.state.copy()
        elif isinstance(state, SparseDM):
            self.state = state
        elif isinstance(state, int):
            self.state = SparseDM(state)
        else:
            raise ValueError(f"Cannot initialize SparseDM from {state}")

        # Optionally override qubit labels
        if qubit_labels is not None:
            assert len(qubit_labels) == self.state.no_qubits, (
                f"Wrong number of qubit labels ({len(qubit_labels)}) ",
                f"provided (expected {self.state.no_qubits})",
            )

            name_map = {k: v for k, v in zip(self.state.names, qubit_labels)}

            self.state.names = [name_map[n] for n in self.state.names]
            self.state.classical = {
                name_map[k]: v for k, v in self.state.classical.items()
            }
