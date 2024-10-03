""":class:``QSimQuantumState`` definition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence, Collection
import random
from typing import ClassVar, TypeAlias, TypeVar

import numpy as np

from loqs.backends import GateRep
from loqs.backends.model.basemodel import InstrumentRep
from loqs.backends.state import BaseQuantumState, OutcomeDict


try:
    from quantumsim.sparsedm import SparseDM as _SparseDM
except ImportError as e:
    raise ImportError("Failed import, cannot use QuantumSim as backend") from e


T = TypeVar("T", bound="QSimQuantumState")

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
        seed: int | None = None,
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

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __str__(self) -> str:
        s = f"Physical {self.name} state:\n"
        s += f"  SparseDM state on {self.state.no_qubits} qubits"
        s += f" ([{self.state.names[0]},...,{self.state.names[-1]}])\n"  # type: ignore
        return s

    def __hash__(self) -> int:
        return hash((hash(self._state), self.seed))

    def apply_reps_inplace(
        self, reps: Sequence, reset_mcms: bool = True
    ) -> OutcomeDict:
        outcomes: OutcomeDict = defaultdict(list)

        for rep, qubits, reptype in reps:
            if reptype == GateRep.QSIM_SUPEROPERATOR:
                if len(qubits) == 1:
                    self.state.apply_ptm(qubits[0], rep)
                elif len(qubits) == 2:
                    # The qubits are flipped here, and this is a known QuantumSim bug
                    self.state.apply_two_ptm(qubits[1], qubits[0], rep)
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
                    # (unless the bit is classical, in which case,
                    # there is only one entry in results)

                    # TODO: At this point, we also have probabilities
                    # We could do also save that data, maybe helpful later

                    m = self._rng.random()
                    if m < results[0][1]:
                        cbit = results[0][0][qbit]
                    else:
                        cbit = results[1][0][qbit]
                    if qbit in self.state.idx_in_full_dm:
                        self.state.project_measurement(qbit, cbit)
                    if reset_mcms:
                        self.state.set_bit(qbit, 0)
                    self.state.renormalize()
                    outcomes[qbit].append(cbit)
            else:
                raise NotImplementedError(f"Cannot apply reptype {reptype}")

        return outcomes

    def apply_reps(
        self, reps: Sequence, reset_mcms: bool = True
    ) -> tuple[QSimQuantumState, OutcomeDict]:
        return super().apply_reps(reps, reset_mcms)

    def copy(self) -> QSimQuantumState:
        return QSimQuantumState(self.state.copy())

    @classmethod
    def _from_serialization(cls: type[T], state: Mapping) -> T:
        qubit_labels = ["qubit_labels"]
        obj = cls(len(qubit_labels), qubit_labels)

        # Restore internal QuantumSim state
        obj.state.classical = state["_qsim_classical"]
        obj.state.idx_in_full_dm = state["_qsim_idx_in_full_dm"]

        dm_class = cls._deserialize_class(
            state["_qsim_dm_class"], check_is_subclass=False
        )
        dm_no_qubits = state["_qsim_dm_no_qubits"]
        dm_data = cls._deserialize_mx(state["_qsim_dm_data"])
        obj.state.full_dm = dm_class(dm_no_qubits, data=dm_data)

        obj.state.max_bits_in_full_dm = state["_qsim_max_bits_in_full_dm"]
        obj.state.classical_probability = state["_qsim_classical_probability"]
        ptm_data = cls.deserialize(state["_qsim_single_ptms_to_do"])
        assert isinstance(ptm_data, dict)
        single_ptms_to_do = defaultdict(list)
        for k, v in ptm_data.items():
            single_ptms_to_do[k] = v
        obj.state.single_ptms_to_do = single_ptms_to_do

        obj.state._last_majority_vote_mask = state["_qsim_maj_vot_mask"]
        obj.state._last_majority_vote_array = cls._deserialize_mx(
            state["_qsim_maj_vot_array"]
        )
        return obj

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()

        # Not saved often, but definitely worth caching when they are
        if (
            hash_to_serial_id_cache is not None
            and hash(self) in hash_to_serial_id_cache
        ):
            state.update(
                {
                    "type": "cached_object_reference",
                    "cache_id": hash_to_serial_id_cache[hash(self)],
                }
            )
            return state

        state.update(
            {
                "qubit_labels": self.state.names,
                "_qsim_classical": self.state.classical,
                "_qsim_idx_in_full_dm": self.state.idx_in_full_dm,
                "_qsim_dm_class": self._serialize_class(
                    type(self.state.full_dm)
                ),
                "_qsim_dm_no_qubits": self.state.full_dm.no_qubits,
                "_qsim_dm_data": self._serialize_mx(
                    self.state.full_dm.to_array()
                ),
                "_qsim_max_bits_in_full_dm": self.state.max_bits_in_full_dm,
                "_qsim_classical_probability": self.state.classical_probability,
                "_qsim_single_ptms_to_do": self.serialize(
                    self.state.single_ptms_to_do
                ),
                "_qsim_maj_vot_mask": self.state._last_majority_vote_mask,
                "_qsim_maj_vot_array": self._serialize_mx(
                    self.state._last_majority_vote_array
                ),
            }
        )

        # If we have a cache, add this to the cache
        if hash_to_serial_id_cache is not None:
            hash_to_serial_id_cache[hash(self)] = len(hash_to_serial_id_cache)
            state.update(
                {
                    "type": "cached_object_source",
                    "cache_id": hash_to_serial_id_cache[hash(self)],
                }
            )

        return state
