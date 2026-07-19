#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections.abc import Sequence
import functools
import itertools
from types import NoneType
import warnings

import numpy as np

from loqs.backends.reps.base import (
    OperationRep,
    RepConstructionError,
    StimCircuitPayloadMixin,
    _num_qubits,
    _resolve_qubits,
)
from loqs.types import Float, NDArray

TP_CHECK_TOL = 1e-8
"""Numerical tolerance for the [](api:KrausGateRep) trace-preservation check."""


class GateRep(OperationRep):
    """Abstract base class for gate operation representations."""

    @classmethod
    def _from_decoded_attrs(cls, attr_dict):
        # `GateRep` is only ever the *recorded* class for a value serialized
        # under the legacy `RepTuple` enum-based format (only concrete leaf
        # classes are instantiable, so current code never produces a
        # serialized object whose recorded class is this abstract base).
        # Seeing `class: "GateRep"` in a file unambiguously identifies a
        # legacy-format value, so we return a legacy tag instead of trying
        # to instantiate the ABC. See
        # `loqs.backends.reps.legacy.RepTuple._from_decoded_attrs`, which
        # consumes this tag (alongside the paired `rep`/`qubits`) to build
        # the correct concrete class. Deferred import avoids a circular
        # dependency, since `legacy.py` itself imports from this module.
        if cls is GateRep:
            from loqs.backends.reps.legacy import _LegacyGateRepValue

            return _LegacyGateRepValue(attr_dict["value"])
        return super()._from_decoded_attrs(attr_dict)


class UnitaryGateRep(GateRep):
    """Unitary matrix representation for a gate.

    `unitary` should be an array with shape `(2^n, 2^n)` where `n` is the
    number of qubits.
    """

    unitary: NDArray

    _SERIALIZE_ATTRS = ["unitary", "qubits"]

    def __init__(
        self, unitary: NDArray, qubits: str | int | Sequence[str | int] = ()
    ) -> None:
        super().__init__(qubits)
        self.unitary = unitary

    @classmethod
    def matches(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
    ) -> bool:
        """Check whether `raw` is a bare array shaped like a unitary.

        Without a known qubit count, this structurally overlaps with
        [](api:PTMGateRep) and [](api:QSimSuperoperatorGateRep): a bare
        array cannot be unambiguously classified as a unitary, a
        Pauli-transfer matrix, or a QuantumSim-basis superoperator by shape
        alone. When `qubits` *is* known (not `None`), the qubit count
        resolves this decisively: a unitary on `n` qubits has shape
        `(2**n, 2**n)`, while a PTM/QSim-superoperator has shape
        `(4**n, 4**n)` -- these are disjoint for any `n`. Callers that
        don't have a qubit count yet (e.g. [](api:DictNoiseModel)'s
        name-only dict entries) resolve the remaining ambiguity by
        choosing one of these three classes explicitly, rather than
        relying on `matches`/[](api:convert)'s generic search.
        """
        if not isinstance(raw, np.ndarray):
            return False
        n = _num_qubits(qubits)
        if n is None:
            return True
        return raw.shape == (2**n, 2**n)

    @classmethod
    def from_raw(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
        **kwargs,
    ) -> "UnitaryGateRep":
        if not cls.matches(raw, qubits):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a numpy array)"
            )
        return cls(raw, _resolve_qubits(qubits))


class PTMGateRep(GateRep):
    """Pauli-transfer matrix representation for a gate.

    A process matrix in the Pauli-product basis. `ptm` should be an array
    with shape `(4^n, 4^n)` where `n` is the number of qubits.
    """

    ptm: NDArray

    _SERIALIZE_ATTRS = ["ptm", "qubits"]

    def __init__(
        self, ptm: NDArray, qubits: str | int | Sequence[str | int] = ()
    ) -> None:
        super().__init__(qubits)
        self.ptm = ptm

    @classmethod
    def matches(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
    ) -> bool:
        """Check whether `raw` is a bare array shaped like a process matrix.

        See [](api:UnitaryGateRep.matches) for why this structurally
        overlaps with [](api:UnitaryGateRep) (resolved decisively when
        `qubits` is known) and [](api:QSimSuperoperatorGateRep) (not
        resolved by shape at all -- both are `(4**n, 4**n)`).
        """
        if not isinstance(raw, np.ndarray):
            return False
        n = _num_qubits(qubits)
        if n is None:
            return True
        return raw.shape == (4**n, 4**n)

    @classmethod
    def from_raw(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
        **kwargs,
    ) -> "PTMGateRep":
        if not cls.matches(raw, qubits):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a numpy array)"
            )
        return cls(raw, _resolve_qubits(qubits))


class QSimSuperoperatorGateRep(GateRep):
    """QuantumSim-basis superoperator representation for a gate.

    Process matrices in QuantumSim's non-standard basis. `superop` should be
    an array with shape `(4^n, 4^n)` where `n` is the number of qubits.
    """

    superop: NDArray

    _SERIALIZE_ATTRS = ["superop", "qubits"]

    def __init__(
        self, superop: NDArray, qubits: str | int | Sequence[str | int] = ()
    ) -> None:
        super().__init__(qubits)
        self.superop = superop

    @classmethod
    def matches(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
    ) -> bool:
        """Check whether `raw` is a bare array shaped like a process matrix.

        See [](api:UnitaryGateRep.matches) for why this structurally
        overlaps with [](api:UnitaryGateRep) (resolved decisively when
        `qubits` is known) and [](api:PTMGateRep) (not resolved by shape
        at all -- both are `(4**n, 4**n)`).
        """
        if not isinstance(raw, np.ndarray):
            return False
        n = _num_qubits(qubits)
        if n is None:
            return True
        return raw.shape == (4**n, 4**n)

    @classmethod
    def from_raw(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
        **kwargs,
    ) -> "QSimSuperoperatorGateRep":
        if not cls.matches(raw, qubits):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a numpy array)"
            )
        return cls(raw, _resolve_qubits(qubits))


class StimCircuitGateRep(StimCircuitPayloadMixin, GateRep):
    """STIM circuit string representation for a gate.

    `circuit_str` should be a STIM circuit string with placeholder qubit
    labels. The string can include both gates (e.g. `"H"`, `"CX"`) and noise
    specifications (e.g. `"X_ERROR(<rate>)"`, `"DEPOLARIZE1(<rate>)"`).
    However, this should not include measurement or reset gates; for those,
    use [](api:StimCircuitInstrumentRep) instead.

    Qubit labels are placeholders indexing into [](api:OperationRep.qubits).

    See [](api:StimCircuitPayloadMixin) for the shared storage/construction
    logic this class shares with [](api:StimCircuitInstrumentRep).
    """


class ProbabilisticStimGateRep(GateRep):
    """A weighted set of STIM circuit strings representing a gate.

    By default, STIM can only do Pauli noise channels. However, some error
    channels can be "unraveled" into a probabilistic choice from Pauli
    channels. For example, amplitude damping can be performed as a
    probabilistic reset.

    `operations` should be a sequence of 2-tuples, where the first entry is
    the circuit string to apply if chosen and the second entry is the
    probability of sampling that operation. Probabilities should be
    positive and add to 1.
    """

    operations: tuple[tuple[str, Float], ...]

    _SERIALIZE_ATTRS = ["operations", "qubits"]

    def __init__(
        self,
        operations: Sequence[tuple[str, Float]],
        qubits: str | int | Sequence[str | int] = (),
    ) -> None:
        super().__init__(qubits)
        self.operations = tuple(tuple(el) for el in operations)  # type: ignore[misc]

    @classmethod
    def matches(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
    ) -> bool:
        """Check that `raw` is a nonempty sequence of `(str, float)` 2-tuples."""
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            return False
        if len(raw) == 0:
            return False
        for el in raw:
            if not isinstance(el, (tuple, list)):
                return False
            if len(el) != 2:
                return False
            if not isinstance(el[0], str):
                return False
            if not isinstance(el[1], (float, np.floating, int)):
                return False
        return True

    @classmethod
    def from_raw(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
        **kwargs,
    ) -> "ProbabilisticStimGateRep":
        if not cls.matches(raw, qubits):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a nonempty "
                "sequence of (str, float) pairs)"
            )
        return cls(raw, _resolve_qubits(qubits))  # type: ignore[arg-type]


class KrausGateRep(GateRep):
    r"""A list of Kraus operators representing a gate.

    The Kraus operators for a CP channel \( \Lambda \) are defined as
    \( K_i \) s.t.

    \[
    \Lambda(\rho) = \sum_i K_i \rho K_i^\dagger
    \]

    The Kraus operators do not have to be unitary, Hermitian, or invertible,
    but the map is also TP if they obey

    \[
    \sum_i K_i^\dagger K_i = I
    \]

    This representation is convenient for all sorts of "unraveling"
    techniques. Critically, it is also possible to unravel non-unital
    channels such as amplitude damping. In that case, one must sample from
    the probability distribution given by

    \[
    P_i = \mathrm{Tr}\left[\rho K_i^\dagger K_i]
    \]

    Note that in the case that \( K_i \) is a scaled unitary, this
    probability will be independent of the state and be a fixed coefficient
    (which is often pulled out of the Kraus matrix in most descriptions of
    the formalism).

    After sampling which Kraus operator to apply, the final state is then

    \[
    \rho \rightarrow K_i \rho K_i^\dagger / P_i
    \]

    Note the renormalization by probability here, since this version of the
    formalism folds the probability into the Kraus matrix, and thus works
    even when the probability is state-dependent.

    This unraveling of non-unital channels can even be done with a
    [](api:STIMQuantumState), enabling fast stabilizer simulation with
    amplitude damping.

    `kraus_operators` should be a sequence of 2-tuples with the first entry
    as an array of size `(2^n, 2^n)` where `n` is the number of qubits, and
    the second entry as a float between 0 and 1 for pre-computed
    probabilities (or `None` in the case of non-unital/state-dependent
    Kraus operators). Even when pre-computed probabilities are provided,
    Kraus operators should not be normalized, i.e. they should include the
    probability also.
    """

    kraus_operators: tuple[tuple[NDArray, Float | None], ...]

    _SERIALIZE_ATTRS = ["kraus_operators", "qubits"]

    def __init__(
        self,
        kraus_operators: Sequence[tuple[NDArray, Float | None]],
        qubits: str | int | Sequence[str | int] = (),
    ) -> None:
        super().__init__(qubits)
        self.kraus_operators = tuple(tuple(el) for el in kraus_operators)  # type: ignore[misc]

    @classmethod
    def matches(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
    ) -> bool:
        """Check that `raw` is a nonempty sequence of `(ndarray, float | None)` pairs.

        This is a purely structural check; it does not perform the
        trace-preservation check (see [](api:KrausGateRep.from_raw)), since
        `matches` may be called speculatively on candidates that ultimately
        aren't selected (e.g. by [](api:convert)), and emitting a warning
        for a rejected candidate would be misleading.
        """
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            return False
        if len(raw) == 0:
            return False
        for el in raw:
            if not isinstance(el, (tuple, list)):
                return False
            if len(el) != 2:
                return False
            if not isinstance(el[0], np.ndarray):
                return False
            if not isinstance(el[1], (float, np.floating, NoneType)):
                return False
        return True

    @classmethod
    def from_raw(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] | None = None,
        tp_check_abstol: Float = TP_CHECK_TOL,
        **kwargs,
    ) -> "KrausGateRep":
        """Construct a [](api:KrausGateRep) from a raw sequence of Kraus operators.

        Parameters
        ----------
        raw:
            A sequence of `(kraus_operator, probability)` pairs.

        qubits:
            Qubit label(s) this operation acts upon, or `None` if not yet
            known.

        tp_check_abstol:
            Absolute tolerance for a trace-preservation check performed on
            the supplied Kraus operators. If finite (the default), a
            [](api:UserWarning) is emitted when
            \\( \\sum_i K_i^\\dagger K_i \\) deviates from the identity by
            more than this tolerance. Set to `float("inf")` to skip the
            check.
        """
        if not cls.matches(raw, qubits):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a nonempty "
                "sequence of (ndarray, float | None) pairs)"
            )
        assert isinstance(raw, Sequence)
        if np.isfinite(tp_check_abstol):
            ops = [K @ K.conj().T for K, _ in raw]
            diff = np.zeros(ops[0].shape, complex)
            diff = np.sum(ops, out=diff, axis=0)
            diff[np.diag_indices_from(diff)] -= 1.0
            if np.any(np.abs(diff) > tp_check_abstol):
                warnings.warn(
                    'Supplied "Kraus operators" do not constitute a TP channel.'
                )
        return cls(raw, _resolve_qubits(qubits))  # type: ignore[arg-type]

    @classmethod
    def from_pauli_stochastic(
        cls,
        rates: Sequence[Float],
        qubits: str | int | Sequence[str | int],
    ) -> "KrausGateRep":
        """Construct a Pauli-stochastic [](api:KrausGateRep) from per-Pauli rates.

        Parameters
        ----------
        rates:
            The probability of each `n`-qubit Pauli string, in
            `itertools.product("IXYZ", repeat=n)` order (`n` inferred from
            `qubits`). Must be nonnegative and sum to 1.

        qubits:
            The targeted qubits.

        Returns
        -------
        KrausGateRep
            The Pauli-stochastic Kraus representation. Terms with
            negligible probability (`< 1e-10`) are omitted.
        """
        normalized_qubits: tuple[str | int, ...] = (
            (qubits,) if isinstance(qubits, (str, int)) else tuple(qubits)
        )
        n = len(normalized_qubits)
        assert all(0 <= p <= 1 for p in rates)
        assert np.isclose(sum(rates), 1)
        assert len(rates) == 4**n

        paulis_1q = {
            "I": np.eye(2),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }

        kraus_reps = []
        for prob, pauli_str in zip(rates, itertools.product("IXYZ", repeat=n)):
            if prob < 1e-10:
                # Skip this term
                continue
            pauli_nq = functools.reduce(
                np.kron, [paulis_1q[c] for c in pauli_str]
            )
            kraus_reps.append((np.sqrt(prob) * pauli_nq, prob))

        return cls(kraus_reps, qubits)

    @classmethod
    def from_depolarizing(
        cls, rate: Float, qubits: str | int | Sequence[str | int]
    ) -> "KrausGateRep":
        """Construct a depolarizing [](api:KrausGateRep).

        A convenience wrapper around [](api:KrausGateRep.from_pauli_stochastic).

        Parameters
        ----------
        rate:
            The depolarizing rate.

        qubits:
            The targeted qubits.

        Returns
        -------
        KrausGateRep
            The depolarizing Kraus representation.
        """
        normalized_qubits: tuple[str | int, ...] = (
            (qubits,) if isinstance(qubits, (str, int)) else tuple(qubits)
        )
        n = 4 ** len(normalized_qubits)
        return cls.from_pauli_stochastic(
            [1 - (n - 1) / n * rate] + [rate / n] * (n - 1), qubits
        )

    @classmethod
    def from_amplitude_damping(
        cls, prob: Float, qubit: str | int
    ) -> "KrausGateRep":
        """Construct a single-qubit amplitude-damping [](api:KrausGateRep).

        Parameters
        ----------
        prob:
            Probability of damping.

        qubit:
            The targeted qubit.

        Returns
        -------
        KrausGateRep
            The amplitude-damping Kraus representation.
        """
        assert 0 <= prob <= 1
        a0 = np.array([[1, 0], [0, np.sqrt(1 - prob)]])
        a1 = np.array([[0, np.sqrt(prob)], [0, 0]])
        return cls([(a0, None), (a1, None)], [qubit])

    def compose(self, other: "GateRep", dedup: bool = True) -> "KrausGateRep":
        r"""Compose this Kraus channel with another, applied afterward.

        Essentially foils the two channels' Kraus operators out:

        .. math:

            M_{i,j} = \sum_i \sum_j c_i c_j K_i L_j

        where `K` are `self`'s Kraus operators, `L` are `other`'s, and `M`
        is the combined output channel (applying `self` first, then
        `other`).

        Parameters
        ----------
        other:
            The [](api:KrausGateRep) or [](api:UnitaryGateRep) to apply
            after `self`. Must act on the same `qubits`.

        dedup:
            Whether (`True`, default) or not to deduplicate the output
            Kraus channel via [](api:KrausGateRep.dedup) (falling back to
            the undeduplicated result if deduplication isn't possible,
            e.g. for non-unital operators with no fixed probabilities).

        Returns
        -------
        KrausGateRep
            The composed channel.
        """
        if not isinstance(other, (KrausGateRep, UnitaryGateRep)):
            raise TypeError(
                f"Cannot compose {type(self).__name__} with "
                f"{type(other).__name__}; expected a KrausGateRep or "
                "UnitaryGateRep"
            )
        assert self.qubits == other.qubits

        def _as_kraus_ops(rep):
            if isinstance(rep, UnitaryGateRep):
                return [(rep.unitary, 1.0)]
            return list(rep.kraus_operators)

        self_ops = _as_kraus_ops(self)
        other_ops = _as_kraus_ops(other)

        # Foil out the terms; probabilities multiply, if available.
        new_kraus_reps = []
        for self_k in self_ops:
            for other_k in other_ops:
                new_k = other_k[0] @ self_k[0]
                try:
                    new_prob = self_k[1] * other_k[1]
                except TypeError:
                    # One was None, we can't compute the probability
                    new_prob = None
                new_kraus_reps.append((new_k, new_prob))

        composed = KrausGateRep(new_kraus_reps, self.qubits)
        if not dedup:
            return composed
        try:
            return composed.dedup()
        except ValueError:
            # Failed to dedup, just return undeduped version
            return composed

    def dedup(self) -> "KrausGateRep":
        """Deduplicate this [](api:KrausGateRep).

        Normalizes all the Kraus operators and checks for any duplicates
        (the same operator up to phase). If duplicates are found, the
        entries are consolidated into a single Kraus operator with the
        combined magnitude of all duplicates.

        Returns
        -------
        KrausGateRep
            The deduplicated Kraus representation.

        Raises
        ------
        ValueError
            If any Kraus operator has no fixed probability (`None`) --
            deduplicating non-unital operators without known probabilities
            isn't currently supported.
        """
        kraus_ops = list(self.kraus_operators)

        # Need to think about how to dedup non-unitaries with no fixed probabilities
        if not all(k[1] is not None for k in kraus_ops):
            raise ValueError(
                "Cannot deduplicate non-unital Kraus operators currently"
            )

        n = len(self.qubits)
        # This will be a list of [normalized K, total summed probability]
        # entries. Unnormalized non-unital Kraus operators without a
        # state-independent probability would need a third column here;
        # for unital operators (the only case handled currently, per the
        # check above), the magnitude and the probability coincide.
        normalized_kraus_ops: list[list] = []

        def _dedup_one(krep):
            k_normed = krep[0] / np.sqrt(krep[1])
            for entry in normalized_kraus_ops:
                # Check the same up to phase
                if np.isclose(np.abs(np.vdot(k_normed, entry[0])), 2**n):
                    # Same as an existing (normalized) Kraus matrix --
                    # update its accumulated probability instead of adding
                    # a new entry.
                    entry[1] += krep[1]
                    return
            normalized_kraus_ops.append([k_normed, krep[1]])

        for krep in kraus_ops:
            _dedup_one(krep)

        # Unnormalize the resulting unique Kraus matrices
        deduped_kraus_reps = [
            (entry[0] * np.sqrt(entry[1]), entry[1])
            for entry in normalized_kraus_ops
        ]

        return KrausGateRep(deduped_kraus_reps, self.qubits)
