#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
import functools
import inspect
import itertools

import numpy as np

from loqs.backends.reps.base import OperationRep, RepConstructionError
from loqs.backends.reps.gatereps import (
    GateRep,
    KrausGateRep,
    ProbabilisticStimGateRep,
    PTMGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.backends.reps.instrumentreps import (
    StimCircuitInstrumentRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.types import Float, NDArray

# `stim` is an optional dependency (soft-dependency idiom already used by
# `stimcircuit.py`/`stimstate.py`) -- the `UnitaryGateRep <-> StimCircuitGateRep`
# edge below is only registered if it's importable, since (unlike the rest
# of this module, which is pure numpy) it genuinely needs `stim.Tableau`.
try:
    import stim
except ImportError:
    stim = None  # type: ignore

_UNITARY_CHECK_TOL = 1e-8
"""Numerical tolerance for checking whether a PTM/Kraus operator is unitary."""

_CHOI_EIGENVALUE_TOL = 1e-9
"""Numerical tolerance below which a Choi-matrix eigenvalue is treated as zero
(i.e. not a genuine Kraus term) during PTM -> Kraus decomposition."""


#####################################################################################################################
# Standard gate unitaries and N-qubit Pauli/QuantumSim basis machinery.
#
# These are needed internally by the pure-numpy conversions below, which
# reimplement the relevant basis-conversion/Kraus-decomposition math
# directly rather than depending on pyGSTi (an optional dependency
# elsewhere in LoQS) for it, and are also exposed publicly as
# `STANDARD_GATE_UNITARIES` so callers (e.g. the example codepacks) don't
# need to hand-copy the same well-known matrices a second time.
#####################################################################################################################

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_S_DAG = _S.conj().T
_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_CZ = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
)

STANDARD_GATE_UNITARIES: dict[str, NDArray] = {
    "I": _I,
    "X": _X,
    "Y": _Y,
    "Z": _Z,
    "H": _H,
    "S": _S,
    "S_DAG": _S_DAG,
    "CX": _CX,
    "CZ": _CZ,
}
"""Well-known single/two-qubit gate unitaries, keyed by generic name.

Reused internally to build the N-qubit Pauli basis (`X`/`Y`/`Z`/`I`) that
this module's own `Unitary<->PTM`/`Kraus<->PTM`/Choi-decomposition math is
built on, and exposed publicly so callers needing a small, fixed set of
standard gates (e.g. example codepacks constructing an ideal noise model)
have one source of truth instead of hand-copying these matrices again.
"""

_PAULI_1Q = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


@functools.lru_cache(maxsize=None)
def _pauli_basis(n: int) -> tuple[NDArray, ...]:
    """The (unnormalized) N-qubit Pauli basis, as `4**n` `(2**n, 2**n)` matrices.

    Ordered as `itertools.product("IXYZ", repeat=n)`, matching pyGSTi's own
    `"pp"` (Pauli-product) basis ordering.
    """
    return tuple(
        functools.reduce(np.kron, [_PAULI_1Q[c] for c in s])
        for s in itertools.product("IXYZ", repeat=n)
    )


_QSIM_1Q = (
    np.array([[1.0, 0], [0, 0]], dtype=complex),  # |0><0|
    np.array([[0, 1], [1, 0]], dtype=complex) / np.sqrt(2),  # X / sqrt(2)
    np.array([[0, -1], [1, 0]], dtype=complex) * 1j / np.sqrt(2),  # Y / sqrt(2)
    np.array([[0, 0], [0, 1]], dtype=complex),  # |1><1|
)


@functools.lru_cache(maxsize=None)
def _qsim_basis(n: int) -> tuple[NDArray, ...]:
    """The N-qubit QuantumSim basis, as `4**n` `(2**n, 2**n)` matrices.

    Only defined for 1-2 qubits, matching [](api:QSimSuperopGateRep)'s
    existing scope (`pygstimodel.py`'s own `_get_gate_rep` already refuses
    to construct one for more than 2 qubits).
    """
    if n not in (1, 2):
        raise RepConstructionError(
            f"The QuantumSim basis is only defined for 1-2 qubits, got {n}"
        )
    return tuple(
        functools.reduce(np.kron, combo)
        for combo in itertools.product(_QSIM_1Q, repeat=n)
    )


def _change_basis(
    ptm: NDArray, from_basis: Sequence[NDArray], to_basis: Sequence[NDArray]
) -> NDArray:
    """Change a superoperator's basis via a similarity transform.

    Reproduces `pygsti.tools.basistools.change_basis` for two explicit,
    linearly-independent Hermitian operator bases of the same dimension
    (e.g. the Pauli basis and the QuantumSim basis): builds the coefficient
    matrix `T` expanding each `to_basis` element in `from_basis`, then
    returns `T^-1 @ ptm @ T`.
    """
    n_basis = len(from_basis)
    T = np.zeros((n_basis, n_basis), dtype=complex)
    for j, Aj in enumerate(from_basis):
        norm2_j = np.vdot(Aj, Aj)
        for i, Bi in enumerate(to_basis):
            T[j, i] = np.vdot(Aj, Bi) / norm2_j
    return np.linalg.inv(T) @ np.asarray(ptm, dtype=complex) @ T


def _choi_kraus_operators(ptm: NDArray, n: int) -> list[NDArray]:
    """Decompose a Pauli-basis PTM into Kraus operators via its Choi matrix.

    Reproduces `pygsti.modelmembers.operations.DenseOperator(ptm, "pp",
    evotype).kraus_operators` (a Choi-Jamiolkowski eigendecomposition) in
    pure numpy: builds the Choi matrix by applying the channel to each
    computational-basis matrix unit `|a><b|`, diagonalizes it, and turns
    each eigenvector with a non-negligible eigenvalue into a Kraus operator
    `sqrt(eigenvalue) * eigenvector.reshape(d, d).T` (the transpose here is
    load-bearing -- verified numerically against amplitude damping, whose
    Kraus operators are not symmetric, unlike e.g. depolarizing noise or
    CNOT, which happened to round-trip correctly even without it).
    """
    d = 2**n
    paulis = _pauli_basis(n)
    ptm = np.asarray(ptm, dtype=complex)

    def _apply(matrix: NDArray) -> NDArray:
        coeffs = np.array([np.trace(Pj @ matrix) / d for Pj in paulis])
        out_coeffs = ptm @ coeffs
        return sum(c * P for c, P in zip(out_coeffs, paulis))

    choi = np.zeros((d * d, d * d), dtype=complex)
    for a in range(d):
        for b in range(d):
            e_ab = np.zeros((d, d), dtype=complex)
            e_ab[a, b] = 1.0
            choi[a * d : (a + 1) * d, b * d : (b + 1) * d] = _apply(e_ab)

    evals, evecs = np.linalg.eigh(choi)
    kraus_ops = []
    for val, vec in zip(evals, evecs.T):
        if val > _CHOI_EIGENVALUE_TOL:
            kraus_ops.append(np.sqrt(val) * vec.reshape(d, d).T)
    return kraus_ops


#####################################################################################################################
# GateRep <-> GateRep pairwise numeric conversions.
#####################################################################################################################


def _unitary_to_ptm(rep: UnitaryGateRep) -> PTMGateRep:
    """Reproduces `pygsti.tools.unitary_to_pauligate`."""
    n = len(rep.qubits)
    d = 2**n
    U = np.asarray(rep.unitary, dtype=complex)
    paulis = _pauli_basis(n)
    ptm = np.array(
        [
            [np.trace(Pi @ U @ Pj @ U.conj().T) / d for Pj in paulis]
            for Pi in paulis
        ]
    )
    return PTMGateRep(ptm, rep.qubits)


def _kraus_to_ptm(rep: KrausGateRep) -> PTMGateRep:
    """Reproduces `pygsti.tools.unitary_to_pauligate`, summed over each
    Kraus operator in `rep`."""
    n = len(rep.qubits)
    d = 2**n
    kraus_ops = [np.asarray(K, dtype=complex) for K, _ in rep.kraus_operators]
    paulis = _pauli_basis(n)
    ptm = np.array(
        [
            [
                sum(np.trace(Pi @ K @ Pj @ K.conj().T) for K in kraus_ops) / d
                for Pj in paulis
            ]
            for Pi in paulis
        ]
    )
    return PTMGateRep(ptm, rep.qubits)


def _unitary_to_kraus(rep: UnitaryGateRep) -> KrausGateRep:
    """Trivial: a unitary is a single Kraus operator with probability 1."""
    return KrausGateRep([(rep.unitary, 1.0)], rep.qubits, tp_check_abstol=None)


def _ptm_to_kraus(rep: PTMGateRep) -> KrausGateRep:
    """Decomposes a Pauli-basis PTM into Kraus operators via `_choi_kraus_operators`.

    Additionally precomputes a fixed probability for any Kraus term that
    is proportional to a unitary once its own scale is divided out (always
    true for a state-independent/classical-probability term), letting
    state backends skip recomputing that probability at simulation time.
    A term that is not proportional to a unitary (a state-dependent,
    non-unital contribution) is left with probability `None`, to be
    computed at simulation time instead.
    """
    n = len(rep.qubits)
    kraus_ops = _choi_kraus_operators(rep.ptm, n)
    if not kraus_ops:
        raise RepConstructionError(
            f"{rep.ptm!r} has no valid Kraus decomposition "
            "(not a completely-positive map)"
        )
    kraus_reps = []
    for K in kraus_ops:
        kkdag = K @ K.conj().T
        prob = kkdag[0, 0]
        if not np.isclose(prob, 0) and np.all(
            np.isclose(kkdag / prob, np.eye(kkdag.shape[0]))
        ):
            # K, once its own scale (prob) is divided out, is the
            # identity -- i.e. K is a scaled unitary with a fixed,
            # state-independent probability.
            kraus_reps.append((K, abs(prob.real)))
        else:
            # Not a scaled unitary, so store None (signal state backends
            # to compute the (state-dependent) probability on the fly).
            kraus_reps.append((K, None))
    return KrausGateRep(kraus_reps, rep.qubits, tp_check_abstol=None)


def _ptm_to_unitary(
    rep: PTMGateRep, unitarity_check_abstol: Float | None = _UNITARY_CHECK_TOL
) -> UnitaryGateRep:
    """Extract a unitary from a PTM.

    A channel's Pauli-basis PTM has exactly one Kraus term (via
    `_choi_kraus_operators`) if and only if the channel is unitary -- e.g.
    Pauli gates and CNOT are unitary and produce exactly one term, while
    depolarizing noise and amplitude damping are not and produce more.
    This structural check always applies; `unitarity_check_abstol` only
    controls whether that single term is also required to be *literally*
    unitary (`None` skips it, e.g. for a projector-like operator that's
    only known to be the map's sole significant term).
    """
    n = len(rep.qubits)
    kraus_ops = _choi_kraus_operators(rep.ptm, n)
    if len(kraus_ops) != 1:
        raise RepConstructionError(
            f"PTM does not correspond to a single unitary operation (found "
            f"{len(kraus_ops)} significant Kraus terms); cannot convert to "
            "UnitaryGateRep"
        )
    U = kraus_ops[0]
    d = 2**n
    if unitarity_check_abstol is not None and not np.allclose(
        U.conj().T @ U, np.eye(d), atol=unitarity_check_abstol
    ):
        raise RepConstructionError(
            "PTM's single Kraus term is not unitary; cannot convert to "
            "UnitaryGateRep"
        )
    return UnitaryGateRep(U, rep.qubits)


def _kraus_to_unitary(
    rep: KrausGateRep, unitarity_check_abstol: Float | None = _UNITARY_CHECK_TOL
) -> UnitaryGateRep:
    """Only succeeds for a single Kraus operator; see `_ptm_to_unitary` for
    `unitarity_check_abstol`."""
    if len(rep.kraus_operators) != 1:
        raise RepConstructionError(
            f"KrausGateRep has {len(rep.kraus_operators)} Kraus operators, "
            "not a single unitary one; cannot convert to UnitaryGateRep"
        )
    K = np.asarray(rep.kraus_operators[0][0], dtype=complex)
    if unitarity_check_abstol is not None and not np.allclose(
        K.conj().T @ K, np.eye(K.shape[0]), atol=unitarity_check_abstol
    ):
        raise RepConstructionError(
            "KrausGateRep's single operator is not unitary; cannot convert "
            "to UnitaryGateRep"
        )
    return UnitaryGateRep(K, rep.qubits)


def _ptm_to_qsim_superoperator(rep: PTMGateRep) -> QSimSuperopGateRep:
    """Changes a PTM from the Pauli basis to the QuantumSim basis via `_change_basis`."""
    n = len(rep.qubits)
    result = _change_basis(rep.ptm, _pauli_basis(n), _qsim_basis(n))
    return QSimSuperopGateRep(result, rep.qubits)


def _qsim_superoperator_to_ptm(rep: QSimSuperopGateRep) -> PTMGateRep:
    """The inverse of `_ptm_to_qsim_superoperator`."""
    n = len(rep.qubits)
    result = _change_basis(rep.superop, _qsim_basis(n), _pauli_basis(n))
    return PTMGateRep(result, rep.qubits)


#####################################################################################################################
# UnitaryGateRep <-> StimCircuitGateRep (only registered if `stim` is
# importable -- see the module-level `try/except ImportError` above).
#
# `endian="big"` is the correct, self-consistent choice for both
# directions, matching LoQS's own multi-qubit convention, where
# `rep.qubits[0]` is the most-significant/leftmost tensor factor (e.g.
# `STANDARD_GATE_UNITARIES["CX"]` treats `qubits[0]` as the control): it's
# the endianness under which `stim.Tableau.from_unitary_matrix`/
# `.to_unitary_matrix` round-trip correctly *and* under which a STIM
# circuit's own qubit indices (e.g. `"CX 0 1"`, control first) already
# match LoQS's placeholder-index convention with no reordering needed.
#####################################################################################################################

_STIM_ENDIAN = "big"


def _unitary_to_stim_circuit(rep: UnitaryGateRep) -> StimCircuitGateRep:
    """Only succeeds if `rep.unitary` is exactly a Clifford operation.

    Uses `stim.Tableau.from_unitary_matrix`, which raises `ValueError` for
    any non-Clifford unitary (e.g. a `T` gate), with no
    approximate/nearest-Clifford fallback. Insensitive to global phase.
    The emitted circuit is a valid decomposition of `rep.unitary`, but not
    necessarily the most obvious-looking one (e.g. a bare `X` gate may come
    out as `H;S;S;H`) -- behaviorally correct regardless, the same
    non-uniqueness situation as Kraus decompositions.
    """
    if stim is None:
        raise RepConstructionError(
            "Converting a UnitaryGateRep to a StimCircuitGateRep requires "
            "the optional `stim` dependency (`pip install loqs[stim]`)"
        )
    try:
        tableau = stim.Tableau.from_unitary_matrix(
            np.asarray(rep.unitary), endian=_STIM_ENDIAN
        )
    except ValueError as e:
        raise RepConstructionError(
            f"{rep.unitary!r} is not exactly a Clifford operation; cannot "
            "convert to StimCircuitGateRep"
        ) from e
    circuit_str = str(tableau.to_circuit())
    return StimCircuitGateRep(circuit_str, rep.qubits)


def _stim_circuit_to_unitary(rep: StimCircuitGateRep) -> UnitaryGateRep:
    """The inverse of `_unitary_to_stim_circuit`.

    Only succeeds if the circuit contains exclusively deterministic,
    unitary Clifford gates -- `stim.Circuit.to_tableau()` natively raises
    `ValueError` for a circuit containing measurement or noise operations,
    which is exactly the condition needed here; no hand-rolled
    per-gate-name lookup table is needed at all, and this works for *any*
    named STIM Clifford gate, not just the small set in
    `STANDARD_GATE_UNITARIES`.
    """
    if stim is None:
        raise RepConstructionError(
            "Converting a StimCircuitGateRep to a UnitaryGateRep requires "
            "the optional `stim` dependency (`pip install loqs[stim]`)"
        )
    n = len(rep.qubits)
    indices = " ".join(str(i) for i in range(n))
    # Prepend a no-op `I <indices>` line so STIM recognizes exactly `n`
    # qubits even if `circuit_str` doesn't happen to reference all of them
    # (e.g. an idle/no-op circuit_str for a declared-but-untouched qubit).
    padded_circuit_str = f"I {indices}\n{rep.circuit_str}" if n else rep.circuit_str
    try:
        circuit = stim.Circuit(padded_circuit_str)
        tableau = circuit.to_tableau()
    except ValueError as e:
        raise RepConstructionError(
            f"{rep.circuit_str!r} does not have a well-defined unitary "
            "Tableau (it contains a measurement or noisy operation); "
            "cannot convert to UnitaryGateRep"
        ) from e
    unitary = tableau.to_unitary_matrix(endian=_STIM_ENDIAN)
    return UnitaryGateRep(unitary, rep.qubits)


#####################################################################################################################
# InstrumentRep <-> InstrumentRep pairwise conversions.
#
# Scoped to the Z-basis family (plus StimCircuitInstrumentRep, below) --
# there's no general "different mathematical encodings of the same
# physical operation" relationship across every InstrumentRep the way
# there is for GateRep, so e.g. StimCircuitInstrumentRep <->
# ZBasisPrePostInstrumentRep is not attempted directly (only via
# ZBasisProjectionInstrumentRep, see the STIM section below).
#####################################################################################################################


def _is_identity_gaterep(rep: GateRep) -> bool:
    """Check whether `rep` represents the identity operation (up to global
    phase). Only recognizes [](api:UnitaryGateRep) -- other `GateRep`
    flavors that happen to represent an identity channel (e.g. a
    single-term `KrausGateRep`) are not attempted, matching this module's
    general policy of not chaining an implicit extra conversion hop inside
    a single pairwise converter.
    """
    if not isinstance(rep, UnitaryGateRep):
        return False
    unitary = np.asarray(rep.unitary)
    d = unitary.shape[0]
    if np.isclose(unitary[0, 0], 0):
        return False
    return np.allclose(unitary, unitary[0, 0] * np.eye(d))


def _zbasis_projection_to_zbasis_pre_post(
    rep: ZBasisProjectionInstrumentRep,
) -> ZBasisPrePostInstrumentRep:
    """Add identity `pre_op`/`post_op` `GateRep`s -- always succeeds."""
    d = 2 ** len(rep.qubits)
    pre_op = UnitaryGateRep(np.eye(d), rep.qubits)
    post_op = UnitaryGateRep(np.eye(d), rep.qubits)
    return ZBasisPrePostInstrumentRep(
        rep.reset, rep.include_outcome, pre_op, post_op, rep.qubits
    )


def _zbasis_pre_post_to_zbasis_projection(
    rep: ZBasisPrePostInstrumentRep,
) -> ZBasisProjectionInstrumentRep:
    """Only succeeds if `pre_op`/`post_op` are both (effectively) identity."""
    if not (
        _is_identity_gaterep(rep.pre_op) and _is_identity_gaterep(rep.post_op)
    ):
        raise RepConstructionError(
            "ZBasisPrePostInstrumentRep's pre_op/post_op are not both "
            "identity; cannot convert to ZBasisProjectionInstrumentRep"
        )
    return ZBasisProjectionInstrumentRep(rep.reset, rep.include_outcome, rep.qubits)


def _zbasis_projection_to_outcome_operation_dict(
    rep: ZBasisProjectionInstrumentRep,
) -> ZBasisOutcomeOperationDictInstrumentRep:
    """Build an `outcome_ops` dict for a Z-basis projection with optional reset.

    `ZBasisOutcomeOperationDictInstrumentRep.outcome_ops` values are
    generalized measurement (Kraus-like) operators, not literally unitary
    despite being stored via [](api:UnitaryGateRep) containers -- this
    matches `npsvstate.py`'s `_apply_instrument_rep` convention, which
    reads `.unitary` directly as a dense matrix. For a Z-basis projection
    with optional reset, outcome `b`'s operator is the (generally
    non-unitary) matrix `|f(b)><b|`, where `f(b)` is `reset` if resetting,
    else `b` itself (i.e. project onto the measured outcome, then map to
    the reset target if one was requested).

    Only supported for exactly 1 qubit, matching the current limitation
    of `npsvstate.py`/`qsimstate.py`'s own `ZBasisOutcomeOperationDictInstrumentRep`
    handling (both raise `NotImplementedError` beyond 1 qubit).
    """
    if len(rep.qubits) != 1:
        raise RepConstructionError(
            "ZBasisOutcomeOperationDictInstrumentRep is only supported for "
            f"exactly 1 qubit, got {len(rep.qubits)}"
        )

    def _outcome_operator(b: int) -> UnitaryGateRep:
        target = b if rep.reset is None else rep.reset
        matrix = np.zeros((2, 2))
        matrix[target, b] = 1.0
        return UnitaryGateRep(matrix, rep.qubits)

    outcome_ops = {0: _outcome_operator(0), 1: _outcome_operator(1)}
    return ZBasisOutcomeOperationDictInstrumentRep(
        outcome_ops, rep.include_outcome, rep.qubits
    )


def _extract_permutation_entry(matrix: NDArray) -> tuple[int, int] | None:
    """If `matrix` has exactly one nonzero entry, of magnitude 1, return its
    `(row, col)` index; otherwise `None`."""
    nonzero = np.argwhere(~np.isclose(matrix, 0))
    if len(nonzero) != 1:
        return None
    i, j = (int(x) for x in nonzero[0])
    if not np.isclose(abs(matrix[i, j]), 1):
        return None
    return i, j


def _outcome_operation_dict_to_zbasis_projection(
    rep: ZBasisOutcomeOperationDictInstrumentRep,
) -> ZBasisProjectionInstrumentRep:
    """The inverse of `_zbasis_projection_to_outcome_operation_dict`.

    Only succeeds if `outcome_ops` is exactly `{0: |f(0)><0|, 1: |f(1)><1|}`
    for some Z-basis-projection-with-optional-reset target function `f`
    (i.e. `f(0) == 0 and f(1) == 1` for no reset, or `f(0) == f(1)` for a
    reset to that value) -- any other `outcome_ops` shape (a genuinely
    different generalized instrument, e.g. a POVM not corresponding to a
    simple projection) is not attempted.
    """
    if len(rep.qubits) != 1:
        raise RepConstructionError(
            "ZBasisOutcomeOperationDictInstrumentRep is only supported for "
            f"exactly 1 qubit, got {len(rep.qubits)}"
        )
    if set(rep.outcome_ops.keys()) != {0, 1}:
        raise RepConstructionError(
            "outcome_ops must have exactly outcomes {0, 1} to convert to "
            "ZBasisProjectionInstrumentRep"
        )

    targets = {}
    for b, op in rep.outcome_ops.items():
        if not isinstance(op, UnitaryGateRep):
            raise RepConstructionError(
                f"outcome_ops[{b}] is not a UnitaryGateRep; cannot convert "
                "to ZBasisProjectionInstrumentRep"
            )
        entry = _extract_permutation_entry(np.asarray(op.unitary))
        if entry is None or entry[1] != b:
            raise RepConstructionError(
                f"outcome_ops[{b}] is not a simple |target><{b}| projector; "
                "cannot convert to ZBasisProjectionInstrumentRep"
            )
        targets[b] = entry[0]

    if targets[0] == 0 and targets[1] == 1:
        reset = None
    elif targets[0] == targets[1]:
        reset = targets[0]
    else:
        raise RepConstructionError(
            "outcome_ops does not correspond to a Z-basis projection with "
            "optional reset (its targets are neither the identity nor a "
            "single fixed reset value)"
        )
    return ZBasisProjectionInstrumentRep(reset, rep.include_outcome, rep.qubits)


#####################################################################################################################
# ZBasisProjectionInstrumentRep <-> StimCircuitInstrumentRep.
#####################################################################################################################

_STIM_SINGLE_LINE_PROJECTIONS: dict[str, tuple[int | None, bool]] = {
    "M": (None, True),
    "MZ": (None, True),
    "MR": (0, True),
    "MRZ": (0, True),
    "R": (0, False),
    "RZ": (0, False),
}


def _zbasis_projection_to_stim_circuit(
    rep: ZBasisProjectionInstrumentRep,
) -> StimCircuitInstrumentRep:
    """Map `(reset, include_outcome)` onto STIM's M/MR/R measurement family.

    `StimCircuitInstrumentRep`'s own docstring documents the exact
    correspondence. Per `stimstate.py`'s outcome-extraction logic, any
    STIM command starting with `M` unconditionally records its outcome --
    so `(None, False)` ("project, but report nothing") has no STIM
    representation and must fail. `reset=1` has no direct STIM
    command either, and is composed as a reset-to-0 measurement (or bare
    reset) followed by a flip.
    """
    n = len(rep.qubits)
    indices = " ".join(str(i) for i in range(n))
    reset, include_outcome = rep.reset, rep.include_outcome

    if reset is None:
        if not include_outcome:
            raise RepConstructionError(
                "(reset=None, include_outcome=False) has no STIM "
                "representation -- every STIM measurement-type command "
                "unconditionally records its outcome"
            )
        circuit_str = f"M {indices}"
    else:
        # reset is constructor-validated to None/0/1, so this is {0, 1}.
        base_command = "MR" if include_outcome else "R"
        circuit_str = f"{base_command} {indices}"
        if reset == 1:
            # A single "X <indices>" line applies X independently to each
            # target, exactly like the "MR"/"R" line above -- no need for
            # one "X" line per qubit.
            circuit_str = f"{circuit_str}\nX {indices}"

    return StimCircuitInstrumentRep(circuit_str, rep.qubits)


def _stim_circuit_to_zbasis_projection(
    rep: StimCircuitInstrumentRep,
) -> ZBasisProjectionInstrumentRep:
    """The inverse of `_zbasis_projection_to_stim_circuit`.

    Unlike the `GateRep` reverse edge (`_stim_circuit_to_unitary`), STIM
    has no native "recognize this as a Z-basis projection" API -- `(reset,
    include_outcome)` is a LoQS-level abstraction, not a STIM concept --
    so this recognizes only the exact forms the forward direction itself
    produces (or hand-written equivalents), not a general STIM-measurement
    circuit parser.
    """
    n = len(rep.qubits)
    expected_targets = [str(i) for i in range(n)]
    lines = [line for line in rep.circuit_str.split("\n") if line.strip()]

    def _parse(line: str) -> tuple[str, list[str]]:
        parts = line.split()
        return parts[0], parts[1:]

    if len(lines) == 1:
        command, targets = _parse(lines[0])
        if command in _STIM_SINGLE_LINE_PROJECTIONS and targets == expected_targets:
            reset, include_outcome = _STIM_SINGLE_LINE_PROJECTIONS[command]
            return ZBasisProjectionInstrumentRep(
                reset, include_outcome, rep.qubits
            )
    elif len(lines) == 2:
        command0, targets0 = _parse(lines[0])
        command1, targets1 = _parse(lines[1])
        if (
            targets0 == expected_targets
            and targets1 == expected_targets
            and command1 == "X"
            and command0 in ("MR", "R")
        ):
            return ZBasisProjectionInstrumentRep(
                1, command0 == "MR", rep.qubits
            )

    raise RepConstructionError(
        f"{rep.circuit_str!r} does not match a recognized "
        "ZBasisProjectionInstrumentRep pattern"
    )


#####################################################################################################################
# Conversion registry and multi-hop dispatch.
#####################################################################################################################

_CONVERTERS: dict[
    tuple[type[OperationRep], type[OperationRep]],
    Callable[..., OperationRep],
] = {
    (UnitaryGateRep, PTMGateRep): _unitary_to_ptm,
    (KrausGateRep, PTMGateRep): _kraus_to_ptm,
    (UnitaryGateRep, KrausGateRep): _unitary_to_kraus,
    (PTMGateRep, KrausGateRep): _ptm_to_kraus,
    (PTMGateRep, UnitaryGateRep): _ptm_to_unitary,
    (KrausGateRep, UnitaryGateRep): _kraus_to_unitary,
    (PTMGateRep, QSimSuperopGateRep): _ptm_to_qsim_superoperator,
    (QSimSuperopGateRep, PTMGateRep): _qsim_superoperator_to_ptm,
    (ZBasisProjectionInstrumentRep, ZBasisPrePostInstrumentRep): (
        _zbasis_projection_to_zbasis_pre_post
    ),
    (ZBasisPrePostInstrumentRep, ZBasisProjectionInstrumentRep): (
        _zbasis_pre_post_to_zbasis_projection
    ),
    (ZBasisProjectionInstrumentRep, ZBasisOutcomeOperationDictInstrumentRep): (
        _zbasis_projection_to_outcome_operation_dict
    ),
    (ZBasisOutcomeOperationDictInstrumentRep, ZBasisProjectionInstrumentRep): (
        _outcome_operation_dict_to_zbasis_projection
    ),
    (ZBasisProjectionInstrumentRep, StimCircuitInstrumentRep): (
        _zbasis_projection_to_stim_circuit
    ),
    (StimCircuitInstrumentRep, ZBasisProjectionInstrumentRep): (
        _stim_circuit_to_zbasis_projection
    ),
}
"""Registry of pairwise `OperationRep` converters, keyed by `(source_cls,
target_cls)`. Consumed by `convert`'s multi-hop shortest-path search.
`UnitaryGateRep <-> StimCircuitGateRep` is added below, conditionally on
`stim` being importable.
"""

if stim is not None:
    _CONVERTERS[(UnitaryGateRep, StimCircuitGateRep)] = _unitary_to_stim_circuit
    _CONVERTERS[(StimCircuitGateRep, UnitaryGateRep)] = _stim_circuit_to_unitary


def _shortest_path(
    source_cls: type[OperationRep], target_cls: type[OperationRep]
) -> list[type[OperationRep]] | None:
    """Breadth-first search for the shortest chain of registered converters
    from `source_cls` to `target_cls`, minimizing the number of hops.

    Returns the sequence of classes to pass through (including `source_cls`
    and `target_cls`), or `None` if no path exists.
    """
    if source_cls is target_cls:
        return [source_cls]

    visited = {source_cls}
    queue: deque[list[type[OperationRep]]] = deque([[source_cls]])
    while queue:
        path = queue.popleft()
        current = path[-1]
        for src_cls, dst_cls in _CONVERTERS:
            if src_cls is not current or dst_cls in visited:
                continue
            new_path = path + [dst_cls]
            if dst_cls is target_cls:
                return new_path
            visited.add(dst_cls)
            queue.append(new_path)
    return None


_ALL_CONCRETE_REPS: tuple[type[OperationRep], ...] = (
    UnitaryGateRep,
    PTMGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    ProbabilisticStimGateRep,
    KrausGateRep,
    ZBasisProjectionInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisOutcomeOperationDictInstrumentRep,
    StimCircuitInstrumentRep,
)
"""Every concrete `GateRep`/`InstrumentRep` leaf class, used by `convert` to
resolve a raw payload's starting class when it doesn't directly match any
of the requested targets (see `convert`'s docstring)."""


def _accepted_kwargs(func: Callable, kwargs: dict) -> dict:
    """Filter `kwargs` down to what `func` accepts by name, so `convert`
    can forward the same `**kwargs` to every candidate class/converter
    without knowing which ones each accepts."""
    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _try_construct(
    cls: type[OperationRep],
    source: object,
    qubits: str | int | Sequence[str | int] | None,
    kwargs: dict,
) -> OperationRep | None:
    """Attempt `cls(source, qubits=qubits, **kwargs)`, returning `None`
    instead of raising if `cls` is abstract, or construction raises
    [](api:RepConstructionError) or `TypeError` (e.g. a class like
    [](api:ZBasisPrePostInstrumentRep) needs several distinct required
    arguments, so a single `source` can never satisfy it)."""
    if inspect.isabstract(cls):
        return None
    try:
        return cls(source, qubits=qubits, **_accepted_kwargs(cls.__init__, kwargs))
    except (RepConstructionError, TypeError):
        return None


def convert(
    source: object,
    target: type[OperationRep] | Sequence[type[OperationRep]],
    qubits: str | int | Sequence[str | int] | None = None,
    **kwargs,
) -> OperationRep:
    """Convert `source` (a raw payload or an `OperationRep`) to `target`.

    1. If `source` already is (or is an instance of) `target`, return it.
    2. Else try constructing each entry of `target` directly from
       `source`, in order; return the first that succeeds.
    3. Else resolve `source`'s unique concrete starting class and hop
       through `_CONVERTERS` to the closest target. A raw payload
       matching more than one concrete class is never guessed at here --
       construct the intended representation explicitly instead.

    Parameters
    ----------
    source:
        A raw, unwrapped payload, or an already-constructed
        [](api:OperationRep) instance.

    target:
        The desired class, or a priority-ordered sequence of candidates.

    qubits:
        Qubit label(s) this operation acts upon, if known, else `None`.

    **kwargs:
        Forwarded to whichever candidate class's constructor accepts them
        (e.g. [](api:KrausGateRep)'s `tp_check_abstol`), and to each
        pairwise converter used along a hop path that accepts them (e.g.
        `_ptm_to_unitary`/`_kraus_to_unitary`'s `unitarity_check_abstol`).

    Returns
    -------
    OperationRep
        The converted (or passed-through) representation.

    Raises
    ------
    RepConstructionError
        If `source` cannot be resolved to a starting class, or no
        conversion path exists from that starting class to any entry in
        `target`.
    """
    targets: tuple[type[OperationRep], ...] = (
        (target,) if isinstance(target, type) else tuple(target)
    )

    if isinstance(source, OperationRep):
        if isinstance(source, targets):
            return source
        source_cls: type[OperationRep] = type(source)
        source_rep: OperationRep = source
    else:
        # Raw payload: try a direct construction against `target` first,
        # in priority order -- no hop needed, so a structurally-ambiguous
        # payload resolves to whichever candidate is listed first.
        for candidate_target in targets:
            result = _try_construct(candidate_target, source, qubits, kwargs)
            if result is not None:
                return result

        # No direct target match -- resolve a single, unambiguous starting
        # class among every known concrete rep class before hopping.
        starting_candidates = [
            (cls, rep)
            for cls in _ALL_CONCRETE_REPS
            if (rep := _try_construct(cls, source, qubits, kwargs)) is not None
        ]
        if not starting_candidates:
            raise RepConstructionError(
                f"{source!r} does not match any known rep class (tried "
                f"{[t.__name__ for t in targets]} directly, and every "
                "other concrete rep class as a starting point for hopping)"
            )
        if len(starting_candidates) > 1:
            raise RepConstructionError(
                f"{source!r} matches more than one rep class "
                f"({[c.__name__ for c, _ in starting_candidates]}); "
                "construct the intended representation explicitly instead "
                "of passing a raw payload here (e.g. "
                "PTMGateRep(source, qubits))"
            )
        source_cls, source_rep = starting_candidates[0]

    if isinstance(source_rep, targets):
        return source_rep

    best_path = None
    for candidate_target in targets:
        path = _shortest_path(source_cls, candidate_target)
        if path is not None and (best_path is None or len(path) < len(best_path)):
            best_path = path
    if best_path is None:
        raise RepConstructionError(
            f"No conversion path from {source_cls.__name__} to any of "
            f"{[t.__name__ for t in targets]}"
        )

    result = source_rep
    for step_cls in best_path[1:]:
        converter = _CONVERTERS[(type(result), step_cls)]
        result = converter(result, **_accepted_kwargs(converter, kwargs))
    return result
