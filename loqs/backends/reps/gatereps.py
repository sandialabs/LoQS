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
from types import NoneType
import warnings

import numpy as np

from loqs.backends.reps.base import (
    OperationRep,
    RepConstructionError,
    StimCircuitPayloadMixin,
)
from loqs.types import Float, NDArray

TP_CHECK_TOL = 1e-8
"""Numerical tolerance for the [](api:KrausGateRep) trace-preservation check."""


class GateRep(OperationRep):
    """Abstract base class for gate operation representations."""

    @classmethod
    def _from_decoded_attrs(cls, attr_dict):
        # `GateRep` is only ever the *recorded* class for an old, pre-refactor
        # `Enum`-member value (new code can never produce a serialized object
        # whose recorded class is this literal abstract base, since only
        # concrete leaf classes are instantiable). Seeing `class: "GateRep"`
        # in a file unambiguously means this came from the old `Enum`, so we
        # return a legacy tag instead of trying to instantiate the ABC. See
        # `loqs.backends.reps.legacy.RepTuple._from_decoded_attrs`, which
        # consumes this tag (alongside the paired `rep`/`qubits`) to build
        # the correct new concrete class. Deferred import avoids a circular
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
    def matches(cls, raw: object) -> bool:
        """Check whether `raw` is a bare array.

        Note that this structurally overlaps with [](api:PTMGateRep) and
        [](api:QSimSuperoperatorGateRep): a bare array cannot be
        unambiguously classified as a unitary, a Pauli-transfer matrix, or a
        QuantumSim-basis superoperator by shape alone. Callers that need to
        upgrade a bare array (e.g. [](api:DictNoiseModel)) resolve this
        ambiguity by choosing one of these three classes explicitly, rather
        than relying on `matches`/[](api:upgrade_gate_rep)'s generic search.
        """
        return isinstance(raw, np.ndarray)

    @classmethod
    def from_raw(
        cls, raw: object, qubits: str | int | Sequence[str | int] = (), **kwargs
    ) -> "UnitaryGateRep":
        if not cls.matches(raw):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a numpy array)"
            )
        return cls(raw, qubits)


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
    def matches(cls, raw: object) -> bool:
        """Check whether `raw` is a bare array.

        See [](api:UnitaryGateRep.matches) for why this structurally
        overlaps with [](api:UnitaryGateRep) and
        [](api:QSimSuperoperatorGateRep).
        """
        return isinstance(raw, np.ndarray)

    @classmethod
    def from_raw(
        cls, raw: object, qubits: str | int | Sequence[str | int] = (), **kwargs
    ) -> "PTMGateRep":
        if not cls.matches(raw):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a numpy array)"
            )
        return cls(raw, qubits)


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
    def matches(cls, raw: object) -> bool:
        """Check whether `raw` is a bare array.

        See [](api:UnitaryGateRep.matches) for why this structurally
        overlaps with [](api:UnitaryGateRep) and [](api:PTMGateRep).
        """
        return isinstance(raw, np.ndarray)

    @classmethod
    def from_raw(
        cls, raw: object, qubits: str | int | Sequence[str | int] = (), **kwargs
    ) -> "QSimSuperoperatorGateRep":
        if not cls.matches(raw):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a numpy array)"
            )
        return cls(raw, qubits)


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
    def matches(cls, raw: object) -> bool:
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
        cls, raw: object, qubits: str | int | Sequence[str | int] = (), **kwargs
    ) -> "ProbabilisticStimGateRep":
        if not cls.matches(raw):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a nonempty "
                "sequence of (str, float) pairs)"
            )
        return cls(raw, qubits)  # type: ignore[arg-type]


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
    def matches(cls, raw: object) -> bool:
        """Check that `raw` is a nonempty sequence of `(ndarray, float | None)` pairs.

        This is a purely structural check; it does not perform the
        trace-preservation check (see [](api:KrausGateRep.from_raw)), since
        `matches` may be called speculatively on candidates that ultimately
        aren't selected (e.g. by [](api:upgrade_gate_rep)), and emitting a
        warning for a rejected candidate would be misleading.
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
        qubits: str | int | Sequence[str | int] = (),
        tp_check_abstol: Float = TP_CHECK_TOL,
        **kwargs,
    ) -> "KrausGateRep":
        """Construct a [](api:KrausGateRep) from a raw sequence of Kraus operators.

        Parameters
        ----------
        raw:
            A sequence of `(kraus_operator, probability)` pairs.

        qubits:
            Qubit label(s) this operation acts upon.

        tp_check_abstol:
            Absolute tolerance for a trace-preservation check performed on
            the supplied Kraus operators. If finite (the default), a
            [](api:UserWarning) is emitted when
            \\( \\sum_i K_i^\\dagger K_i \\) deviates from the identity by
            more than this tolerance. Set to `float("inf")` to skip the
            check.
        """
        if not cls.matches(raw):
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
        return cls(raw, qubits)  # type: ignore[arg-type]
