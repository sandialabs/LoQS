#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from functools import singledispatchmethod
import numpy as np
from typing import ClassVar, TypeAlias, TypeVar


from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    KrausGateRep,
    OperationRep,
    UnitaryGateRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
    convert as convert_rep,
    is_rep_compatible,
)
from loqs.backends.state import BaseQuantumState, OutcomeDict
from loqs.internal.serializable import Serializable
from loqs.types import NDArray

T = TypeVar("T", bound="NumpyStatevectorQuantumState")

# Type aliases for static type checking
NumpyStatevectorLike: TypeAlias = (
    "NumpyStatevectorQuantumState | int | np.ndarray | Sequence[int]"
)
"""Types that this backend can cast to an underlying state object."""

QubitTypes: TypeAlias = str | int
"""Types this backend can use for qubit labels.

Note that this is technically not a true restriction,
but keeping it simple as other types are unlikely.
"""

# TODO: Possible performance improvements:
# Use QSim's trick of removing measured qubits for smaller statevector
# Use two 2**N arrays and matmul's out= to prevent new mem allocations
# Exploit that Pauli-stochastic Kraus operators are scaled axis permutations

KRAUS_SAMPLING_MODES = ("lazy", "choice")
"""Supported Kraus sampling modes.

"lazy": inverse-CDF sampling; state-dependent probabilities are computed
only until the sampled operator is reached (fast, default).

"choice": legacy eager sampling; all state-dependent probabilities are
computed up front and ``rng.choice`` draws the operator. Consumes the RNG
stream exactly as LoQS did before lazy sampling existed, so seeded
trajectories from older runs reproduce bit-for-bit.
"""

CONTRACTION_MODES = ("matmul", "einsum")
"""Supported contraction modes for applying operators to the state tensor.

Governs every operator application (unitary gates, Kraus operators, and
measurement projectors), which all route through the same block matvec.

"matmul": moveaxis + a single BLAS matmul + moveaxis (fast, default).

"einsum": the original ``np.einsum`` contraction, preserved verbatim as an
independent reference implementation for equivalence testing and
benchmarking. Amplitudes may differ from "matmul" at machine precision
(floating-point summation order), but not beyond.
"""


def _single_dense_operator(rep: GateRep) -> NDArray:
    """Extract the single dense `(2**n, 2**n)` operator underlying `rep`,
    regardless of its concrete `GateRep` representation. Passes
    `unitarity_check_abstol=None` to skip `convert`'s literal-unitarity
    check, since `rep` may be a projector-like operator (e.g. a
    `ZBasisOutcomeOperationDictInstrumentRep` outcome operator) rather
    than an actually-unitary one; the "exactly one dense term" structural
    check still applies.

    Raises
    ------
    RepConstructionError
        If `rep` doesn't correspond to a single dense operator.
    """
    unitary_rep = convert_rep(rep, UnitaryGateRep, unitarity_check_abstol=None)
    return unitary_rep.unitary


class NumpyStatevectorQuantumState(BaseQuantumState):

    name: ClassVar[str] = "NumPy Statevector"

    _SERIALIZE_ATTRS = [
        "_state",
        "qubit_labels",
        "seed",
        "kraus_sampling",
        "contraction",
        "d",
    ]

    _SERIALIZE_ATTRS_MAP = {"_state": "state"}

    _state: np.ndarray
    """Underlying state object."""

    qubit_labels: list[QubitTypes]
    """Qubit labels."""

    kraus_sampling: str
    """Kraus sampling mode; see [](api:KRAUS_SAMPLING_MODES)."""

    contraction: str
    """Operator contraction mode; see [](api:CONTRACTION_MODES)."""

    d: list[int]
    """List of dimensions of each subsystem (e.g. 2 for qubits, 3 for qutrits)."""

    @property
    def state(self) -> np.ndarray:
        """Get the underlying quantum state vector.

        Returns
        -------
        np.ndarray
            The quantum state vector as a numpy array.
        """
        return self._state

    @property
    def input_reps(self) -> list[type[GateRep | InstrumentRep]]:
        return [
            UnitaryGateRep,
            KrausGateRep,
            ZBasisProjectionInstrumentRep,
            ZBasisPrePostInstrumentRep,
            ZBasisOutcomeOperationDictInstrumentRep,
        ]

    def __init__(
        self,
        state: NumpyStatevectorLike,
        qubit_labels: Sequence[QubitTypes] | None = None,
        seed: int | None = None,
        kraus_sampling: str | None = None,
        contraction: str | None = None,
        d: int | Sequence[int] = 2,
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

        kraus_sampling:
            Kraus sampling mode; see [](api:KRAUS_SAMPLING_MODES). If not
            provided, inherited from `state` when copy-constructing,
            otherwise defaults to "lazy".

        contraction:
            Operator contraction mode; see [](api:CONTRACTION_MODES). If not
            provided, inherited from `state` when copy-constructing,
            otherwise defaults to "matmul".
        """
        self.qubit_labels = []
        self.reset_seed(seed)
        # These may be None here; resolved below
        self.kraus_sampling = kraus_sampling
        self.contraction = contraction

        # Determine number of subsystems
        if qubit_labels is not None:
            num_subsystems = len(qubit_labels)
        elif isinstance(state, NumpyStatevectorQuantumState):
            num_subsystems = len(state.qubit_labels)
        elif isinstance(state, int):
            num_subsystems = state
        elif isinstance(state, np.ndarray):
            if len(state.shape) > 1:
                num_subsystems = len(state.shape)
            else:
                if isinstance(d, int):
                    num_subsystems = int(np.round(np.log2(state.flatten().shape[0])))
                else:
                    num_subsystems = len(d)
        elif isinstance(state, Sequence) and not isinstance(state, str):
            num_subsystems = len(state)
        else:
            raise ValueError(f"Cannot determine number of subsystems from {state}")

        # Resolve dimensions list
        if isinstance(d, int):
            self.d = [d] * num_subsystems
        else:
            self.d = list(d)
        assert len(self.d) == num_subsystems, f"Length of d ({len(self.d)}) must match number of subsystems ({num_subsystems})"

        if isinstance(state, NumpyStatevectorQuantumState):
            self._state = state._state
            self.qubit_labels = state.qubit_labels
            self.seed = state.seed
            self._rng = state._rng
            self.d = state.d
            if kraus_sampling is None:
                self.kraus_sampling = state.kraus_sampling
            if contraction is None:
                self.contraction = state.contraction
        elif isinstance(state, int):
            self._state = np.zeros(tuple(self.d), np.complex128)
            self._state[(0,) * state] = 1
        elif isinstance(state, np.ndarray):
            self._state = state.copy()
            curr_shape = state.shape
            if list(curr_shape) != self.d:
                # Flat or mismatch, reshape to self.d
                self._state = self.state.reshape(tuple(self.d))
        elif isinstance(state, Sequence) and all(
            [el in range(self.d[i]) for i, el in enumerate(state)]
        ):
            self._state = np.zeros(tuple(self.d), np.complex128)
            self._state[*state] = 1
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

        if self.kraus_sampling is None:  # We haven't set it yet
            self.kraus_sampling = "lazy"
        assert self.kraus_sampling in KRAUS_SAMPLING_MODES, (
            f"kraus_sampling must be one of {KRAUS_SAMPLING_MODES}, "
            f"got {self.kraus_sampling}"
        )
        if self.contraction is None:  # We haven't set it yet
            self.contraction = "matmul"
        assert self.contraction in CONTRACTION_MODES, (
            f"contraction must be one of {CONTRACTION_MODES}, "
            f"got {self.contraction}"
        )

    def __str__(self) -> str:
        s = f"Physical {self.name} state (ds={self.d}):\n"
        s += f"  NumPy statevector on {len(self.qubit_labels)} subsystems"
        s += f" ([{self.qubit_labels[0]},...,{self.qubit_labels[-1]}])\n"
        return s

    # Source - https://stackoverflow.com/a/64436208
    def _slice(self, a: np.ndarray, axis, start=None, end=None, step=1):
        assert axis >= -len(a.shape) and axis < len(a.shape)
        return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

    def apply_reps(
        self, reps: Sequence[OperationRep]
    ) -> tuple[NumpyStatevectorQuantumState, OutcomeDict]:
        return super().apply_reps(reps)

    def apply_reps_inplace(self, reps: Sequence[OperationRep]) -> OutcomeDict:
        outcomes: OutcomeDict = defaultdict(list)

        for rep in reps:
            if isinstance(rep, GateRep):
                self._apply_gate_rep(rep)
            elif isinstance(rep, InstrumentRep):
                rep_outcomes = self._apply_instrument_rep(rep)

                # Merge outcomes with already observed outcomes
                for k, v in rep_outcomes.items():
                    outcomes[k].extend(v)
            else:
                raise ValueError(
                    f"Cannot apply unknown rep type {type(rep).__name__}"
                )

        return outcomes

    @singledispatchmethod
    def _apply_gate_rep(self, rep: GateRep) -> None:
        raise NotImplementedError(f"Cannot apply {type(rep).__name__} to {self.name}")

    @_apply_gate_rep.register
    def _(self, rep: UnitaryGateRep) -> None:
        qubits = rep.qubit_labels
        assert len(qubits) > 0
        unitary = rep.unitary

        target_dims = [self.d[self.qubit_labels.index(q)] for q in qubits]
        total_target_dim = int(np.prod(target_dims))
        assert unitary.shape == (total_target_dim, total_target_dim), (
            f"UnitaryGateRep on {qubits} has shape {unitary.shape}, but "
            f"the state's subsystem dimensions {target_dims} expect shape "
            f"({total_target_dim}, {total_target_dim})"
        )

        self._state = self._block_matvec(unitary, qubits, self.state)

    @_apply_gate_rep.register
    def _(self, rep: KrausGateRep) -> None:
        qubits = rep.qubit_labels
        assert len(qubits) > 0

        kraus_operators = rep.kraus_operators
        target_dims = [self.d[self.qubit_labels.index(q)] for q in qubits]
        total_target_dim = int(np.prod(target_dims))
        assert all(
            K.shape == (total_target_dim, total_target_dim)
            for K, _ in kraus_operators
        ), (
            f"KrausGateRep on {qubits} has an operator shape inconsistent "
            f"with the state's subsystem dimensions {target_dims}"
        )

        if self.kraus_sampling == "choice":
            self._apply_kraus_choice(kraus_operators, qubits)
        else:
            self._apply_kraus_lazy(kraus_operators, qubits)

        assert np.isclose(np.linalg.norm(self.state), 1)

    def _apply_kraus_lazy(self, rep, qubits) -> None:
        """Apply a Kraus channel using lazy inverse-CDF sampling.

        A single uniform draw selects the operator: operator i owns the
        interval [C_{i-1}, C_i) of cumulative probabilities, so it is chosen
        with probability exactly p_i regardless of list order. Probabilities
        given as None are computed from the state only until the sampled
        operator is reached, so each Kraus operator is applied at most once
        and (for low-noise channels with the dominant operator first) the
        typical cost is a single matvec.
        """
        assert self._rng is not None
        r = self._rng.random()

        cum = 0.0
        choice = None
        chosen_prob = None
        chosen_Kprod = None
        last_valid = None
        for i, (K, prob) in enumerate(rep):
            Kprod = None
            if prob is None:
                # Compute state-dependent probability
                Kprod = self._block_matvec(K, qubits, self.state)
                prob = np.vdot(Kprod, Kprod).real
            assert prob >= -1e-9
            prob = max(prob, 0.0)
            if prob > 0:
                last_valid = (i, prob, Kprod)
            cum += prob
            if r < cum:
                choice, chosen_prob, chosen_Kprod = i, prob, Kprod
                break

        if choice is None:
            # r landed in the float-roundoff sliver between cum and 1;
            # fall back to the last operator with nonzero probability
            assert last_valid is not None and np.isclose(cum, 1.0)
            choice, chosen_prob, chosen_Kprod = last_valid

        # Normalize final subvector
        if chosen_Kprod is None:
            # Probability was given, so the product was never computed
            chosen_Kprod = self._block_matvec(
                rep[choice][0], qubits, self.state
            )
        self._state = chosen_Kprod / np.sqrt(chosen_prob)

    def _apply_kraus_choice(self, rep, qubits) -> None:
        """Apply a Kraus channel using legacy eager rng.choice sampling.

        All state-dependent probabilities are computed up front (one matvec
        per None-probability operator). Kept because it consumes the RNG
        stream exactly as pre-lazy LoQS did, so old seeded trajectories
        reproduce bit-for-bit.
        """
        Ks = [r[0] for r in rep]
        probs = [r[1] for r in rep]

        # Compute any state-dependent probabilities we need to
        Kprods = {}
        for i, (K, prob) in enumerate(rep):
            if prob is None:
                # Compute probability

                Kprod = self._block_matvec(K, qubits, self.state)
                prob_calc = np.vdot(Kprod, Kprod)
                prob_calc = np.abs(np.real_if_close(prob_calc))
                assert np.isreal(prob_calc)

                probs[i] = prob_calc.real
                Kprods[i] = Kprod

        assert all([prob >= 0 for prob in probs])
        assert np.isclose(sum(probs), 1.0)

        # # Sample
        try:
            choice = self._rng.choice(range(len(rep)), size=1, p=probs)[0]
        except ValueError:
            if np.abs(1 - sum(probs)) < 1e-7:
                renormed_probs = np.asarray(probs) / np.sum(probs)
                choice = self._rng.choice(
                    range(len(rep)), size=1, p=renormed_probs
                )[0]
            else:
                raise ValueError(
                    "Kraus operator probabilities sum to "
                    f"{sum(probs)}, too far from 1 to renormalize"
                )

        # Normalize final subvector
        try:
            chosen_Kprod = Kprods[choice]
        except KeyError:
            # Was not computed for probability, compute it now
            chosen_Kprod = self._block_matvec(Ks[choice], qubits, self.state)
        self._state = chosen_Kprod / np.sqrt(probs[choice])

    def _block_matvec(self, submat, sublbls, vec) -> np.ndarray:
        if self.contraction == "einsum":
            return self._block_matvec_einsum(submat, sublbls, vec)
        return self._block_matvec_matmul(submat, sublbls, vec)

    def _block_matvec_matmul(self, submat, sublbls, vec) -> np.ndarray:
        n_sub = len(sublbls)

        # Axes of the state tensor targeted by the operator
        try:
            axes = [self.qubit_labels.index(lbl) for lbl in sublbls]
        except ValueError as e:
            raise ValueError(
                "Rep's qubit is not in state's qubit labels\n" + str(e)
            )
        target_dims = [self.d[axis] for axis in axes]
        total_target_dim = int(np.prod(target_dims))
        assert len(submat.flat) == total_target_dim ** 2

        # Bring the target axes to the front (in sublbls order, matching the
        # operator's row/column qubit ordering), contract with a single BLAS
        # matmul, then restore the original axis order. Materialize the
        # result contiguously: returning a strided view makes every
        # downstream contraction read badly-ordered memory and is a net loss
        moved = np.moveaxis(vec, axes, range(n_sub))
        out = submat.reshape(total_target_dim, total_target_dim) @ moved.reshape(total_target_dim, -1)
        return np.ascontiguousarray(
            np.moveaxis(out.reshape(moved.shape), range(n_sub), axes)
        )

    def _block_matvec_einsum(self, submat, sublbls, vec) -> np.ndarray:
        # The original einsum contraction, preserved verbatim as a reference
        # implementation; must remain equivalent to _block_matvec_matmul
        n_sub = len(sublbls)
        n_tot = len(vec.shape)

        # We will need n_qubits..n_qubits+n_subqubits temp indices (vals of the dict below)
        # These will map to the qubit labels in our qubit subset (keys of the dict below)
        try:
            sub_idx_map = {
                self.qubit_labels.index(lbl): n_tot + i
                for i, lbl in enumerate(sublbls)
            }
        except ValueError as e:
            raise ValueError(
                "Rep's qubit is not in state's qubit labels\n" + str(e)
            )
        target_dims = [self.d[axis] for axis in sub_idx_map]
        total_target_dim = int(np.prod(target_dims))
        assert len(submat.flat) == total_target_dim ** 2
        submat = submat.reshape(tuple(target_dims) + tuple(target_dims))

        # Get contraction indices
        # Our vector will just have 0..n_qubits-1 indices to start
        vec_in_idxs = list(range(n_tot))
        # Our submatrix has indices of temp labels (rows, output states) and subset labels (cols, input states)
        submat_idxs = list(sub_idx_map.values()) + list(sub_idx_map.keys())

        # Just the start vec, but the sublbls replaced with the temp ones to do the contraction
        vec_out_idxs = [sub_idx_map.get(i, i) for i in range(n_tot)]

        # Now perform the einsum
        return np.einsum(
            vec, vec_in_idxs, submat, submat_idxs, vec_out_idxs, optimize=True
        )

    @singledispatchmethod
    def _apply_instrument_rep(self, rep: InstrumentRep) -> OutcomeDict:
        raise NotImplementedError(
            f"Cannot apply {type(rep).__name__} to {self.name}"
        )

    @_apply_instrument_rep.register
    def _(self, rep: ZBasisProjectionInstrumentRep) -> OutcomeDict:
        qubits = rep.qubit_labels
        assert len(qubits) > 0

        outcomes: OutcomeDict = defaultdict(list)
        reset = rep.reset
        include_outcomes = rep.include_outcome

        # TODO: Could do it all at once probably
        # but currently just copying measureRenormalizeQubit behavior
        for qbit in qubits:
            cbit = self._apply_projective_z_measure(qbit, reset)
            if include_outcomes:
                outcomes[qbit].append(cbit)
        return outcomes

    @_apply_instrument_rep.register
    def _(self, rep: ZBasisPrePostInstrumentRep) -> OutcomeDict:
        qubits = rep.qubit_labels
        assert len(qubits) > 0

        outcomes: OutcomeDict = defaultdict(list)
        reset = rep.reset
        include_outcomes = rep.include_outcome

        # is_rep_compatible is backend-specific, so still checked here.
        preop = rep.pre_op
        postop = rep.post_op
        assert is_rep_compatible(type(preop), self.input_reps)
        assert is_rep_compatible(type(postop), self.input_reps)

        # Apply the pre-op
        self.apply_reps_inplace([preop])

        # Do perfect measurement
        for qbit in qubits:
            cbit = self._apply_projective_z_measure(qbit, reset)
            if include_outcomes:
                outcomes[qbit].append(cbit)

        # Apply the post-op
        self.apply_reps_inplace([postop])
        return outcomes

    @_apply_instrument_rep.register
    def _(self, rep: ZBasisOutcomeOperationDictInstrumentRep) -> OutcomeDict:
        qubits = rep.qubit_labels
        assert len(qubits) > 0

        outcomes: OutcomeDict = defaultdict(list)
        include_outcomes = rep.include_outcome

        if len(qubits) > 1:
            raise NotImplementedError(
                "More than 1-qubit instruments not yet implemented"
            )
        instrument_dict = rep.outcome_ops
        assert set(instrument_dict.keys()) == set((0, 1))

        # Compute the probability of measuring 0
        # (Same as Kraus logic in _apply_gate_rep)
        prod = self._block_matvec(
            _single_dense_operator(instrument_dict[0]), qubits, self.state
        )
        prob_0 = np.vdot(prod, prod)

        # Use RNG to see if we measure 0 or 1
        assert self._rng is not None
        m = self._rng.random()
        cbit = 0 if m < prob_0 else 1
        if include_outcomes:
            outcomes[qubits[0]].append(cbit)

        # Apply the correct PTM based on the classical output we see
        # and renormalize
        if cbit == 0:
            # We already computed this product
            self._state = prod / np.sqrt(prob_0)
        else:
            self._state = self._block_matvec(
                _single_dense_operator(instrument_dict[1]), qubits, self.state
            ) / np.sqrt(1 - prob_0)

        return outcomes

    def _apply_projective_z_measure(self, qbit, reset) -> int:
        target_idx = self.qubit_labels.index(qbit)
        dim = self.d[target_idx]

        probs = []
        for c in range(dim):
            target_slice = self._slice(self.state, target_idx, start=c, end=c+1)
            prob_c = np.vdot(target_slice.flat, target_slice.flat).real
            probs.append(max(prob_c, 0.0))
        
        probs = np.array(probs)
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs = probs / sum_probs
        else:
            probs = np.ones(dim) / dim

        assert self._rng is not None
        cbit = self._rng.choice(dim, p=probs)

        proj_mat = np.zeros((dim, dim), np.complex128)
        if reset is None:
            reset = cbit
        assert reset in range(dim), f"reset must be in range(dim), got {reset}"
        proj_mat[reset, cbit] = 1 / np.sqrt(probs[cbit])

        self._state = self._block_matvec(proj_mat, [qbit], self.state)

        return cbit

    def reset_seed(self, new_seed: int | None) -> None:
        self.seed = new_seed
        self._rng = np.random.default_rng(new_seed)

    def copy(self) -> NumpyStatevectorQuantumState:
        new_state = NumpyStatevectorQuantumState(
            deepcopy(self.state),
            qubit_labels=self.qubit_labels,
            seed=self.seed,
            kraus_sampling=self.kraus_sampling,
            contraction=self.contraction,
            d=self.d,
        )
        new_state._rng = deepcopy(self._rng)
        return new_state

    def print_bitstring_amplitudes(self):
        """Print the amplitudes of all bitstrings in the quantum state.

        This method prints the qubit labels and then iterates through all possible
        bitstrings, displaying those with amplitudes greater than a threshold (1e-6).
        """
        n_qubits = len(self.qubit_labels)
        print(self.qubit_labels)
        total_dim = int(np.prod(self.d))
        for i in range(total_dim):
            temp = i
            idx = [0] * n_qubits
            for j in reversed(range(n_qubits)):
                base = self.d[j]
                idx[j] = temp % base
                temp //= base
            amp = self.state[*idx]
            if np.abs(amp) > 1e-6:
                # Printed digit order is reversed relative to qubit_labels
                # (last qubit's digit printed first), matching this
                # method's original qubit-only convention.
                bs = "".join(str(x) for x in reversed(idx))
                print(f"{bs}: {amp}")
