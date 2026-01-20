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
            self._state = np.zeros((2,) * state, np.complex128)
            self._state[(0,) * state] = 1
        elif isinstance(state, np.ndarray):
            assert NotImplementedError("Have to fix reshape")
            self._state = state.copy()
        elif isinstance(state, Sequence) and all(
            [el in [0, 1] for el in state]
        ):
            self._state = np.zeros((2,) * len(state), np.complex128)
            self._state[state] = 1
        else:
            raise ValueError(
                f"Cannot initialize NumpyStatevectorQuantumState from {state}"
            )

        if qubit_labels is not None:
            self.qubit_labels = list(qubit_labels)
        if (
            len(self.qubit_labels) == 0
        ):  # We haven't set it yet, default to ints
            self.qubit_labels = list(range(len(self.state.shape)))
        assert len(self.qubit_labels) == len(
            self.state.shape
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

    # Source - https://stackoverflow.com/a/64436208
    def _slice(self, a: np.ndarray, axis, start=None, end=None, step=1):
        assert axis >= -len(a.shape) and axis < len(a.shape)
        return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

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
                raise ValueError(f"Cannot apply unknown reptype {reptype}")

        return outcomes

    def _apply_gate_rep(self, reptuple: RepTuple) -> None:
        rep = reptuple.rep

        qubits = reptuple.qubits
        assert isinstance(qubits, (tuple, list)) and len(qubits) > 0

        reptype = reptuple.reptype

        if reptype == GateRep.UNITARY:
            assert isinstance(rep, np.ndarray) and rep.shape == (
                2 * len(qubits),
                2 * len(qubits),
            )

            self._state = self._block_matvec(rep, qubits, self.state)
        elif reptype == GateRep.KRAUS_OPERATORS:
            assert isinstance(rep, (list, tuple))
            assert all([isinstance(mat, np.ndarray) for mat in rep])
            assert all(
                [
                    mat.shape == (2 * len(qubits), 2 * len(qubits))
                    for mat in rep
                ]
            )

            # TODO: We could cache probabilities for unitary Kraus operators, maybe in the model
            # Compute probabilities (have to do this in order for non-unital Kraus operations to work)
            # probs = []
            # Kprods = []
            # for K in rep:
            #     subvec = self._get_subvector(qubits, self.state)
            #     subprob = np.vdot(subvec, subvec)

            #     Kprod = K @ subvec

            #     prob = np.vdot(Kprod, Kprod) / subprob
            #     assert np.isreal(prob)

            #     probs.append(prob.real)
            #     Kprods.append(Kprod)

            # assert np.isclose(np.sum(probs), 1)

            # # Sample
            # assert self._rng is not None
            # choice = self._rng.choice(range(len(rep)), size=1, p=probs)[0]

            # # Normalize final subvector
            # final_subvec = Kprods[choice] / np.sqrt(probs[choice])

            # self._set_subvector(final_subvec, qubits, self.state)
            # assert np.isclose(np.linalg.norm(self.state), 1)
        else:
            raise ValueError(f"Cannot apply GateRep {reptype}")

    def _block_matvec(self, submat, sublbls, vec) -> np.ndarray:
        n_sub = len(sublbls)
        n_tot = len(vec.shape)
        assert len(submat.flat) == 4**n_sub
        submat = submat.reshape((2,) * 2 * n_sub)

        # Get contraction indices
        # Our vector will just have 0..n_qubits-1 indices to start
        vec_in_idxs = list(range(n_tot))

        # We will need n_qubits..n_qubits+n_subqubits temp indices (vals of the dict below)
        # These will map to the qubit labels in our qubit subset (keys of the dict below)
        sub_idx_map = {
            self.qubit_labels.index(lbl): n_tot + i
            for i, lbl in enumerate(sublbls)
        }
        # Our submatrix has indices of subset + temp labels
        submat_idxs = list(sub_idx_map.keys()) + list(sub_idx_map.values())

        # Just the start vec, but the sublbls replaced with the temp ones to do the contraction
        vec_out_idxs = [sub_idx_map.get(i, i) for i in range(n_tot)]

        # Now perform the einsum
        # TODO: For multiple back-to-back mults, it might be better to einsum_path
        return np.einsum(
            vec, vec_in_idxs, submat, submat_idxs, vec_out_idxs, optimize=True
        )

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
            self._state = self._block_matvec(
                rep_to_apply, qubits, self.state
            ) / (np.sqrt(prob_0) if cbit == 0 else np.sqrt(1 - prob_0))

        return outcomes

    def _apply_projective_z_measure(self, qbit, reset) -> int:
        target_idx = self.qubit_labels.index(qbit)

        # Compute probability of measuring 0 on the target qubit
        target_slice = self._slice(self.state, target_idx, end=1)
        prob_0 = np.dot(target_slice.flat, target_slice.flat)

        # Probabilistically select 0 or 1 outcome
        assert self._rng is not None
        cbit = 0 if self._rng.random() < prob_0 else 1

        # Get the projector (I'll wrap normalization into it)
        proj_mat = np.zeros((2, 2), np.complex128)
        if reset is None:
            reset = cbit
        assert reset in [0, 1]
        if cbit == 0:
            # Measuring 0 (normalize by prob 0) and final state is given by reset
            proj_mat[0, reset] = 1 / np.sqrt(prob_0)
        else:
            # Measuring 1 (normalize by prob 1) and final state is given by reset
            proj_mat[1, reset] = 1 / np.sqrt(1 - prob_0)

        # Apply projector
        self._state = self._block_matvec(proj_mat, [qbit], self.state)

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
