""":class:``QSimQuantumState`` definition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence, Collection
import random
from typing import ClassVar, TypeAlias

from loqs.backends import GateRep
from loqs.backends.model.basemodel import InstrumentRep
from loqs.backends.state import BaseQuantumState, OutcomeDict


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


class QSimQuantumState(BaseQuantumState):
    """Base class for an object that holds a QuantumSim SparseDM state."""

    name: ClassVar[str] = "QuantumSim"

    _state: _SparseDM
    """Underlying state object."""

    @property
    def state(self) -> _SparseDM:
        return self._state

    @property
    def input_reps(self) -> list[GateRep | InstrumentRep]:
        return [GateRep.QSIM_SUPEROPERATOR, InstrumentRep.ZBASISPROJECTION]

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

    def apply_reps_inplace(
        self, reps: Sequence, reset_mcms: bool = True
    ) -> OutcomeDict:
        outcomes: OutcomeDict = defaultdict(list)

        for rep, qubits, reptype in reps:
            if reptype == GateRep.QSIM_SUPEROPERATOR:
                if len(qubits) == 1:
                    self.state.apply_ptm(qubits[0], rep)
                elif len(qubits) == 2:
                    self.state.apply_two_ptm(qubits[0], qubits[1], rep)
                else:
                    raise ValueError(
                        "Cannot apply more than a 2 qubit operation"
                    )
            elif reptype == InstrumentRep.ZBASISPROJECTION:
                # TODO: Could do it all at once probably
                # but currently just copying measureRenormalizeQubit behavior
                for qbit in qubits:
                    results = self.state.peak_multiple_measurements([qbit])
                    # Results has following structure
                    # results: [({'A1': 0}, p0), ({'A1': 1}, p1)}
                    # results[a][0][qubit] = measurement value
                    # results[a][1]        = corresponding probability
                    # where a is in {0,1}
                    assert len(results) == 2
                    # TODO: At this point, we also have probabilities
                    # We could do also save that data, maybe helpful later

                    m = random.random()
                    if m < results[0][1]:
                        cbit = results[0][0][qbit]
                    else:
                        cbit = results[1][0][qbit]
                    self.state.project_measurement(qbit, cbit)
                    if reset_mcms:
                        self.state.set_bit(qbit, 0)
                    self.state.renormalize()
                    outcomes[qbit].append(cbit)
            else:
                raise NotImplementedError(f"Cannot apply reptype {reptype}")

        return outcomes

    def apply_reps(
        self, reps: Sequence
    ) -> tuple[QSimQuantumState, OutcomeDict]:
        return super().apply_reps(reps)

    def copy(self) -> QSimQuantumState:
        return QSimQuantumState(self.state.copy())
