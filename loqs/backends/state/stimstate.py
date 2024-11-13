""":class:`.STIMQuantumState` definition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import numpy as np
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends import GateRep
from loqs.backends.model.basemodel import InstrumentRep
from loqs.backends.reps import RepTuple
from loqs.backends.state import BaseQuantumState, OutcomeDict


try:
    from stim import Circuit as _Circuit
    from stim import Tableau as _Tableau
    from stim import TableauSimulator as _TableauSimulator
except ImportError as e:
    raise ImportError("Failed import, cannot use STIM as backend") from e


T = TypeVar("T", bound="STIMQuantumState")

# Type aliases for static type checking
STIMStateCastableTypes: TypeAlias = (
    "STIMQuantumState | _TableauSimulator | _Tableau | int | Sequence[int]"
)
"""Types that this backend can cast to an underlying state object."""

QubitTypes: TypeAlias = str | int
"""Types this backend can use for qubit labels.

Note that this is technically not a true restriction of SparseDM,
but keeping it simple as other types are unlikely.
"""


class STIMQuantumState(BaseQuantumState):
    """Base class for an object that holds a STIM Tableau."""

    name: ClassVar[str] = "STIM Tableau"

    _state: _TableauSimulator
    """Underlying state object."""

    qubit_labels: list[QubitTypes]
    """Qubit labels.

    These are used to map local ints
    to global ints in
    :attr:`.GateRep.STIM_CIRCUIT_STR` reps.
    """

    @property
    def state(self) -> _TableauSimulator:
        return self._state

    @property
    def input_reps(self) -> list[GateRep | InstrumentRep]:
        return [
            GateRep.STIM_CIRCUIT_STR,
            GateRep.KRAUS_OPERATORS,
            InstrumentRep.ZBASIS_PROJECTION,
            InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
        ]

    def __init__(
        self,
        state: STIMStateCastableTypes,
        qubit_labels: Sequence[QubitTypes] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        state:
            A representation of the underlying state. If an integer is passed,
            an all-0 state with that number of qubits is passed

        qubit_labels:
            Optional qubit labels. If not provided, the default range of ints
            is used.
        """
        self.qubit_labels = []
        if isinstance(state, STIMQuantumState):
            # If we are setting a seed here, do not copy internal RNG
            # Otherwise, DO copy internal RNG
            self._state = state._state.copy(copy_rng=seed is None, seed=seed)
            self.qubit_labels = state.qubit_labels
        elif isinstance(state, _TableauSimulator):
            self._state = state.copy(copy_rng=seed is None, seed=seed)
        elif isinstance(state, _Tableau):
            self._state = _TableauSimulator(seed=seed)
            self._state.set_inverse_tableau(state)
        elif isinstance(state, int):
            self._state = _TableauSimulator(seed=seed)
            self._state.set_num_qubits(state)
        elif isinstance(state, Sequence) and all(
            [el in [0, 1] for el in state]
        ):
            self._state = _TableauSimulator(seed=seed)
            self._state.set_num_qubits(len(state))
            # Flip specified bits
            for bit, val in enumerate(state):
                if val:
                    self.state.x(bit)
        else:
            raise ValueError(
                f"Cannot initialize TableauSimulator from {state}"
            )

        if qubit_labels is not None:
            self.qubit_labels = list(qubit_labels)
        if (
            len(self.qubit_labels) == 0
        ):  # We haven't set it yet, default to ints
            self.qubit_labels = list(range(self.state.num_qubits))
        assert (
            len(self.qubit_labels) == self.state.num_qubits
        ), "Must specify a qubit label for every qubit"

        self.reset_seed(seed)

    def __str__(self) -> str:
        s = f"Physical {self.name} state:\n"
        s += f"  STIM state on {self.state.num_qubits} qubits"
        s += f" ([{self.qubit_labels[0]},...,{self.qubit_labels[-1]}])\n"
        return s

    def __hash__(self) -> int:
        return hash(
            (hash(self._state), self.hash(self.qubit_labels), self.seed)
        )

    def apply_reps_inplace(self, reps: Sequence) -> OutcomeDict:
        outcomes: OutcomeDict = defaultdict(list)

        for reptuple in reps:
            reptype = reptuple.reptype
            if isinstance(reptype, GateRep):
                self._apply_gate_rep(reptuple)
            elif isinstance(reptype, InstrumentRep):
                rep_outcomes = self._apply_instrument_rep(reptuple)

                # Merge outcomes with already observed outcomes
                for k, v in rep_outcomes.items():
                    outcomes[k].extend(v)
            else:
                raise NotImplementedError(
                    f"Cannot apply unknown reptype {reptype}"
                )

        return outcomes

    def apply_reps(
        self, reps: Sequence
    ) -> tuple[STIMQuantumState, OutcomeDict]:
        return super().apply_reps(reps)

    def copy(self) -> STIMQuantumState:
        new_state = STIMQuantumState(self.state, self.qubit_labels, self.seed)
        new_state._rng = deepcopy(self._rng)
        return new_state

    def reset_seed(self, new_seed: int | None) -> None:
        # We explicitly don't want to copy RNG here, force a new RNG seed
        self._state = self._state.copy(copy_rng=False, seed=new_seed)
        self.seed = new_seed
        self._rng = np.random.default_rng(new_seed)

    def _apply_gate_rep(self, reptuple: RepTuple):
        rep = reptuple.rep

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and len(qubits) > 0

        reptype = reptuple.reptype
        assert isinstance(reptype, GateRep)

        if reptype == GateRep.STIM_CIRCUIT_STR:
            assert isinstance(rep, str)

            # Create mapping from placeholder to global qubit indices
            local_to_global = {
                str(i): str(self.qubit_labels.index(q))
                for i, q in enumerate(qubits)
            }

            # Split string for easy processing
            mapped_lines = []
            for line in rep.split("\n"):
                if len(line) == 0 or line.startswith("TICK"):
                    # Empty or TICK line, skip
                    continue

                entries = line.split()

                mapped_entries = [entries[0]]  # instruction is unchanged
                mapped_entries += [local_to_global[e] for e in entries[1:]]

                mapped_lines.append(" ".join(mapped_entries))

            mapped_circuit_str = "\n".join(mapped_lines)
            mapped_circuit = _Circuit(mapped_circuit_str)

            self.state.do_circuit(mapped_circuit)
        elif reptype == GateRep.KRAUS_OPERATORS:
            assert isinstance(rep, (list, tuple))
            assert all(isinstance(K, np.ndarray) for K in rep)

            # Get the current state vector in little endian,
            # i.e. entries correspond to 000, 001, 010, 011,
            # 100, 101, 110, 111 for a 3-qubit example
            state_vec = self.state.state_vector(endian="little")

            # We need to compute probabilities as:
            # P_i = \mathrm{Tr}\left[\rho K_i^\dagger K_i]
            # But our rho is a pure state, so we can simplify this to
            # P_i = Tr[<\Psi| K_i^\dagger K_i |\Psi> ] = KPsi * KPsi

            # We will do these computations using QuantumSim-like einsum calls
            # First, we'll reshape the statevec into a tensor and use einsum
            # to contract out the unaffected indices (and reorder if need)
            # Then we can revectorize and simply do a matmul with our Kraus ops
            Nq = len(self.qubit_labels)
            state_vec_tensor = state_vec.reshape(
                [
                    2,
                ]
                * Nq
            )

            # Trace out indices we don't need
            # However, for state vec, these need to add in quadrature, like a density mx would
            # I don't want to expand to a full density mx, but we can square now, trace, and sqrt,
            # i.e. store the diagonals of the density mx. This is fine because we are only tracing
            # over it anyway
            # TODO: Not sure this is correct
            in_indices = list(range(Nq))
            out_indices = [self.qubit_labels.index(q) for q in qubits]
            reduced_state_vec_tensor = np.einsum(
                np.square(state_vec_tensor),
                in_indices,
                out_indices,
                optimize=True,
            )
            # Also reshape to col vector, i.e. |\Psi>
            reduced_state_vec = np.sqrt(reduced_state_vec_tensor).reshape(
                (-1, 1)
            )

            # Now we can just compute our probabilities
            probs = []
            for K in rep:
                KPsi = K @ reduced_state_vec
                probs.append(np.sqrt(np.vdot(KPsi, KPsi)))

            # Pick an operation to sample
            idx_to_apply = np.random.choice(list(range(len(rep))), p=probs)

            # Get rescaled versions of the Kraus operator so that we can
            # perform: \rho \rightarrow K_i \rho K_i^\dagger / P_i
            rescaled_K = rep[idx_to_apply] / np.sqrt(probs[idx_to_apply])

            # Use another einsum to multiply the rescaled Kraus operator through
            # This effectively handles embedding the Kraus operator in the full space
            # Borrows the reversed from QuantumSim, which I think is for efficiency?
            out_indices = list(reversed(range(Nq)))
            temp_indices = list(range(Nq, Nq + len(qubits)))
            qubit_indices = [self.qubit_labels.index(q) for q in qubits]
            in_indices = list(reversed(range(Nq)))
            # Replace incoming indices with temp ones to do the matmul correctly
            for ti, qidx in zip(temp_indices, qubit_indices):
                in_indices[Nq - qidx - 1] = ti
            K_indices = qubit_indices + temp_indices

            # Perform contraction to get new state
            new_state_vec = np.einsum(
                state_vec_tensor,
                in_indices,
                rescaled_K,
                K_indices,
                out_indices,
                optimize=True,
            )

            # Overwrite simulator state (with the flatted state vector)
            self.state.set_state_from_state_vector(
                new_state_vec.ravel(), endian="little"
            )

            # Check that STIM has not snapped our state down
            assert np.allclose(
                self.state.state_vector(), new_state_vec
            ), "State vec changed, likely not a stabilizer state"
        else:
            raise NotImplementedError(f"Cannot apply GateRep {reptype}")

    def _apply_instrument_rep(self, reptuple: RepTuple) -> OutcomeDict:
        rep = reptuple.rep
        assert isinstance(rep, (tuple, list)) and len(rep) > 1
        reset = rep[0]
        include_outcomes = rep[1]

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and len(qubits) > 0

        reptype = reptuple.reptype
        assert isinstance(reptype, InstrumentRep)

        outcomes: OutcomeDict = defaultdict(list)

        if reptype == InstrumentRep.ZBASIS_PROJECTION:
            for qbit in qubits:
                cbit = self._measure_and_reset(qbit, reset)
                if include_outcomes:
                    outcomes[qbit].append(cbit)
        elif reptype == InstrumentRep.ZBASIS_PRE_POST_OPERATIONS:
            # Check we can apply the reps
            preop = rep[2]
            postop = rep[3]
            assert reset in [None, 0, 1]
            assert preop.reptype in self.input_reps
            assert postop.reptype in self.input_reps
            assert isinstance(preop.reptype, GateRep)
            assert isinstance(postop.reptype, GateRep)
            # TODO: Strict subsets is OK too
            assert preop.qubits == qubits
            assert postop.qubits == qubits

            # Apply the pre-op
            self.apply_reps_inplace([preop])

            # Do perfect measurement
            for qbit in qubits:
                cbit = self._measure_and_reset(qbit, reset)
                if include_outcomes:
                    outcomes[qbit].append(cbit)

            # Apply the post-op
            self.apply_reps_inplace([postop])
        else:
            raise NotImplementedError(f"Cannot apply InstrumentRep {reptype}")

        return outcomes

    def _measure_and_reset(
        self, qubit: QubitTypes, reset: int | None = None
    ) -> int:
        qidx = self.qubit_labels.index(qubit)
        cbit = int(self.state.measure(qidx))

        if reset is not None:
            # Reset to 0
            self.state.reset(qidx)

            # Reset to 1, if needed
            if reset == 1:
                self.state.x(qidx)

        return cbit

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        qubit_labels = state["qubit_labels"]
        tableau_circ = _Circuit(state["_stim_tableau_circuit"])
        # TODO: This does not set serialization... not sure I can do much about that
        # Do I recommend doing a copy with a new seed after deserialization?
        return cls(
            _Tableau.from_circuit(tableau_circ), qubit_labels=qubit_labels
        )

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        tableau_circ = self.state.current_inverse_tableau().to_circuit()
        # TODO: Missing RNG!
        state.update(
            {
                "qubit_labels": self.qubit_labels,
                "_stim_tableau_circuit": str(tableau_circ),
            }
        )
        return state
