""":class:`.BaseQuantumState` definition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import itertools
from typing import ClassVar, TypeAlias, TypeVar

import numpy as np

from loqs.backends.model.basemodel import GateRep, InstrumentRep
from loqs.backends.reps import RepTuple
from loqs.backends.state import BaseQuantumState, OutcomeDict


T = TypeVar("T", bound="NumpyStatevectorQuantumState")

# Type aliases for static type checking
NumpyStatevectorCastableTypes: TypeAlias = (
    "NumpyStatevectorQuantumState | int | np.ndarray | Sequence[int]"
)
"""Types that this backend can cast to an underlying state object."""

QubitTypes: TypeAlias = str | int
"""Types this backend can use for qubit labels.

Note that this is technically not a true restriction,
but keeping it simple as other types are unlikely.
"""


class NumpyStatevectorQuantumState(BaseQuantumState):
    """Base class for an object that holds a (physical) quantum state."""

    name: ClassVar[str] = "NumPy Statevector"

    _state: np.ndarray
    """Underlying state object."""

    qubit_labels: list[QubitTypes]
    """Qubit labels."""

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def input_reps(self) -> list[GateRep | InstrumentRep]:
        return [
            GateRep.UNITARY,
            GateRep.KRAUS_OPERATORS,
            InstrumentRep.ZBASIS_PROJECTION,
            InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
        ]

    def __init__(
        self,
        state: NumpyStatevectorCastableTypes,
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

        seed:
            Optional RNG seed. If not provided, default NumPy RNG behavior applies.
        """
        self.qubit_labels = []
        self.seed = None
        self._rng = None

        if isinstance(state, NumpyStatevectorQuantumState):
            self._state = state._state
            self.qubit_labels = state.qubit_labels
            self.seed = state.seed
            self._rng = state._rng
        elif isinstance(state, int):
            self._state = np.zeros((2 * state, 1), complex)
            self._state[::2] = 1
        elif isinstance(state, np.ndarray):
            self._state = state.copy()
        elif isinstance(state, Sequence) and all(
            [el in [0, 1] for el in state]
        ):
            self._state = np.zeros((2 * len(state), 1), complex)
            for i, el in enumerate(state):
                self.state[2 * i + el] = 1
        else:
            raise ValueError(
                f"Cannot initialize NumpyStatevectorQuantumState from {state}"
            )

        if qubit_labels is not None:
            self.qubit_labels = list(qubit_labels)
        if (
            len(self.qubit_labels) == 0
        ):  # We haven't set it yet, default to ints
            self.qubit_labels = list(range(self.state.shape[0] / 2))
        assert (
            len(self.qubit_labels) == self.state.shape[0] / 2
        ), "Must specify a qubit label for every qubit"

        if self.seed is None:
            self.seed = seed
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)

    def __str__(self) -> str:
        s = f"Physical {self.name} state:\n"
        s += f"  NumPy statevector on {self.state.shape[0]} qubits"
        s += f" ([{self.qubit_labels[0]},...,{self.qubit_labels[-1]}])\n"
        return s

    def __hash__(self) -> int:
        return hash(
            (hash(self._state), self.hash(self.qubit_labels), self.seed)
        )

    def apply_reps(
        self, reps: Sequence[RepTuple]
    ) -> tuple[NumpyStatevectorQuantumState, OutcomeDict]:
        return super().apply_reps(reps)

    def apply_reps_inplace(self, reps: Sequence[RepTuple]) -> OutcomeDict:
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

    def _apply_gate_rep(self, reptuple: RepTuple) -> None:
        rep = reptuple.rep
        # TODO: Could do shape checking here

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and len(qubits) > 0

        reptype = reptuple.reptype

        if reptype == GateRep.UNITARY:
            self._block_matvec_inplace(rep, qubits, self.state)
        elif reptype == GateRep.KRAUS_OPERATORS:
            assert isinstance(rep, (list, tuple)) and all(
                [isinstance(mat, np.ndarray) for mat in rep]
            )

            # TODO: We could cache probabilities for unitary Kraus operators, maybe in the model
            # Compute probabilities (have to do this in order for non-unital Kraus operations to work)
            probs = []
            for K in rep:
                # We need to do P_i = Tr(rho K^dag K), but for a pure state rho = |x><x|,
                # this is just <x|K^dag K|x>, or the dot of K|x> with itself
                prod = self._block_matvec(K, qubits, self.state)
                probs.append(np.vdot(prod, prod))

            assert np.isclose(np.sum(probs), 1)

            # Sample
            assert self._rng is not None
            choice = self._rng.choice(range(len(rep)), size=1, p=probs)

            # Apply chosen Kraus operator
            # From rho -> K rho K^dag / P, we have |x> -> K |x> / sqrt(P) for the pure state version
            # Note that we have to normalize by probability since it is folded into K for our formalism
            self._block_matvec_inplace(rep[choice], qubits, self.state)
            self._state /= np.sqrt(probs[choice])
        else:
            raise NotImplementedError(f"Cannot apply GateRep {reptype}")

    def _block_matvec_inplace(self, submat, sublbls, vec) -> None:
        # Map sublabels in qubit label indices
        full_idxs = [self.qubit_labels.index(lbl) for lbl in sublbls]

        # Pull out the appropriate subvector
        subvec = np.zeros((2 * len(sublbls), 1), np.complex128)
        for i, full_idx in enumerate(full_idxs):
            full_idx = full_idxs[i]
            subvec[2 * i : 2 * i + 2] = vec[full_idx : full_idx + 2]

        # Perform dense mat-vec product
        subprod = submat @ subvec

        # Put back into full vector
        for i, full_idx in enumerate(full_idxs):
            vec[full_idx : full_idx + 2] = subprod[2 * i : 2 * i + 2]

    def _block_matvec(self, submat, sublbls, vec: np.ndarray) -> np.ndarray:
        newvec = vec.copy()
        self._block_matvec_inplace(submat, sublbls, newvec)
        return newvec

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
            # TODO: Could do it all at once probably
            # but currently just copying measureRenormalizeQubit behavior
            for qbit in qubits:
                cbit = self._apply_projective_z_measure(qbit, reset)
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
                cbit = self._apply_projective_z_measure(qbit, reset)
                if include_outcomes:
                    outcomes[qbit].append(cbit)

            # Apply the post-op
            self.apply_reps_inplace([postop])
        elif reptype == InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT:
            if len(qubits) > 1:
                raise NotImplementedError(
                    "More than 1-qubit instruments not yet implemented"
                )
            instrument_dict = rep[0]
            assert set(instrument_dict.keys()) == set((0, 1))

            # Compute the probability of measuring 0
            # (Same as Kraus logic in _apply_gate_rep)
            prod = self._block_matvec(instrument_dict[0], qubits, self.state)
            prob_0 = np.vdot(prod, prod)

            # Use RNG to see if we measure 0 or 1
            assert self._rng is not None
            m = self._rng.random()
            cbit = 0 if m < prob_0 else 1
            if include_outcomes:
                outcomes[qubits[0]].append(cbit)

            # Apply the correct PTM based on the classical output we see
            rep_to_apply = instrument_dict[cbit]
            assert rep_to_apply.reptype in self.input_reps
            assert isinstance(rep_to_apply.reptype, GateRep)
            self.apply_reps_inplace([rep_to_apply])

            # Propogate and renormalize (maybe not needed, but safer to do it now)
            self._block_matvec_inplace(rep_to_apply, qubits, self.state)
            self._state /= (
                np.sqrt(prob_0) if cbit == 0 else np.sqrt(1 - prob_0)
            )

        return outcomes

    def _apply_projective_z_measure(self, qbit, reset) -> int:
        # Compute all coefficients for basis states with 0 in the desired qubit
        num_qubits = len(self.qubit_labels)
        qidx = self.qubit_labels.index(qbit)

        prob_0 = 0
        projected_state = np.zeros_like(self.state)
        for i in range(2 * num_qubits):
            # Binary rep gives us state 0 or 1 in standard basis (in big-endian)
            # First, let's check the bit we want and make sure it is low
            if i & 1 << qidx:
                # Bit is high, this is for state 1, skip
                continue

            # Next, convert from int to a list of [1, 0] or [0, 1] as the statevec rep
            sv_list = [
                [1, 0] if i & 1 << j else [0, 1]
                for j in range(num_qubits - 1, -1, -1)
            ]
            # Use itertools to flatten this
            sv_flat = list(itertools.chain.from_iterable(sv_list))
            # And finally cast to a row vector
            basis_vec = np.asarray(sv_flat, np.complex128)
            # And normalize
            basis_vec /= np.linalg.norm(basis_vec)

            # Now compute the coefficient with state
            coeff = np.vdot(basis_vec, self.state)
            print(f"{coeff=}")

            # Add to prob and projected state
            prob_0 += coeff
            projected_state += coeff * basis_vec.reshape((14, 1))

        # Probabilistically select 0 or 1 outcome
        assert self._rng is not None
        cbit = 0 if self._rng.random() < prob_0 else 1

        print(f"DEBUG: {prob_0=}, {cbit=}")

        # If we measured 1, compute the projected state now (if not resetting to 0)
        assert reset in [None, 0, 1]
        if cbit == 1 and reset == 1:
            projected_state = np.zeros_like(self.state)
            for i in range(2 * num_qubits):
                # Binary rep gives us state 0 or 1 in standard basis (in big-endian)
                # First, let's check the bit we want and make sure it is low
                if not (i & 1 << qidx):
                    # Bit is low, this was for state 0, skip
                    continue

                # All same as state 0 computation
                sv_list = [
                    [1, 0] if i & 1 << j else [0, 1]
                    for j in range(num_qubits - 1, -1, -1)
                ]
                sv_flat = itertools.chain.from_iterable(sv_list)
                basis_vec = np.asarray(sv_flat, np.complex128)
                basis_vec /= np.linalg.norm(basis_vec)
                coeff = np.vdot(basis_vec, self.state)

                projected_state += coeff * basis_vec.reshape((14, 1))

        # Set state to post-projection and renormalize
        renorm = np.sqrt(prob_0) if cbit == 0 else np.sqrt(1 - prob_0)
        self._state = projected_state / renorm

        return cbit

    def copy(self) -> NumpyStatevectorQuantumState:
        new_state = NumpyStatevectorQuantumState(
            deepcopy(self.state),
            qubit_labels=self.qubit_labels,
            seed=self.seed,
        )
        new_state._rng = deepcopy(self._rng)
        return new_state

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        statevector = cls.deserialize(state["_statevector"])
        assert isinstance(statevector, np.ndarray)
        qubit_labels = state["qubit_labels"]
        seed = state["seed"]
        return cls(statevector, qubit_labels, seed=seed)

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        state = super()._to_serialization()
        # TODO: RNG. Maybe https://stackoverflow.com/q/63081108
        state.update(
            {
                "qubit_labels": self.qubit_labels,
                "seed": self.seed,
                "_statevector": self.serialize(self._state),
            }
        )
        return state
