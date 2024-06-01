""":class:``QSimQuantumState`` definition.
"""

from __future__ import annotations

from collections.abc import Sequence, Collection
from typing import ClassVar, Literal, TypeAlias

from loqs.backends import OpRep
from loqs.backends.state import BaseQuantumState


try:
    from quantumsim.sparsedm import SparseDM as _SparseDM
except ImportError as e:
    raise ImportError("Failed import, cannot use QuantumSim as backend") from e

# Type aliases for static type checking
CastableTypes: TypeAlias = "QSimQuantumState | int | _SparseDM"
"""Types that this backend can cast to an underlying state object."""

QubitTypes: TypeAlias = str | int
"""Types this backend can use for qubit labels.

Note that this is technically not a true restriction of SparseDM,
but keeping it simple as other types are unlikely.
"""

OpRepInputs: TypeAlias = Literal[OpRep.QSIM_SUPEROPERATOR]
"""OpRep types this backend can take as inputs."""


class QSimQuantumState(BaseQuantumState):
    """Base class for an object that holds a QuantumSim SparseDM state."""

    name: ClassVar[str] = "QuantumSim"

    _state: _SparseDM
    """Underlying state object."""

    @property
    def state(self) -> _SparseDM:
        return self._state

    def __init__(
        self,
        state: CastableTypes,
        qubit_labels: Collection[QubitTypes] | None = None,
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
        if isinstance(state, QSimQuantumState):
            self._state = state._state.copy()
        elif isinstance(state, _SparseDM):
            self._state = state
        elif isinstance(state, int):
            self._state = _SparseDM(state)
        else:
            raise ValueError(f"Cannot initialize SparseDM from {state}")

        # Optionally override qubit labels
        if qubit_labels is not None:
            assert len(qubit_labels) == self.state.no_qubits, (
                f"Wrong number of qubit labels ({len(qubit_labels)}) ",
                f"provided (expected {self.state.no_qubits})",
            )

            # The names field is not well typed in SparseDM
            name_map = {k: v for k, v in zip(self.state.names, qubit_labels)}  # type: ignore

            self.state.names = [name_map[n] for n in self.state.names]  # type: ignore
            self.state.classical = {
                name_map[k]: v for k, v in self.state.classical.items()
            }

    def apply_operator_reps_inplace(self, op_reps: Sequence) -> None:
        # TODO: Instruments/measurements
        for rep, qubits in op_reps:
            if len(qubits) == 1:
                self.state.apply_ptm(qubits[0], rep)
            elif len(qubits) == 2:
                self.state.apply_two_ptm(qubits[0], qubits[1], rep)
            else:
                raise ValueError("Cannot apply more than a 2 qubit operation")

    def apply_operator_reps(self, op_reps: Sequence) -> QSimQuantumState:
        return super().apply_operator_reps(op_reps)

    def copy(self) -> QSimQuantumState:
        return QSimQuantumState(self.state.copy())
