#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Literal

from loqs.backends.reps.base import (
    OperationRep,
    RepConstructionError,
    StimCircuitPayloadMixin,
)
from loqs.backends.reps.gatereps import GateRep


class InstrumentRep(OperationRep):
    """Abstract base class for instrument (mid-circuit measurement) representations."""

    @classmethod
    def _from_decoded_attrs(cls, attr_dict):
        # See `GateRep._from_decoded_attrs` for why this special-cases
        # `cls is InstrumentRep` and defers to
        # `loqs.backends.reps.legacy.RepTuple._from_decoded_attrs`.
        if cls is InstrumentRep:
            from loqs.backends.reps.legacy import _LegacyInstrumentRepValue

            return _LegacyInstrumentRepValue(attr_dict["value"])
        return super()._from_decoded_attrs(attr_dict)


class ZBasisProjectionInstrumentRep(InstrumentRep):
    """Z-basis projection representation for an instrument.

    Essentially a perfect mid-circuit measurement, followed by optional
    reset.
    """

    reset: Literal[None, 0, 1]
    """`None` for no reset, or `0`/`1` for reset to the corresponding state."""

    include_outcome: bool
    """Whether the measurement outcome should be recorded.

    E.g. `reset=0, include_outcome=False` would look like a pure reset.
    """

    _SERIALIZE_ATTRS = ["reset", "include_outcome", "qubits"]

    def __init__(
        self,
        reset: Literal[None, 0, 1],
        include_outcome: bool,
        qubits: str | int | Sequence[str | int] | None = (),
    ) -> None:
        """Construct a [](api:ZBasisProjectionInstrumentRep).

        Raises
        ------
        RepConstructionError
            If `reset` isn't `None`, `0`, or `1`, or `include_outcome`
            isn't `True`/`False` (or, for compatibility with files
            serialized before `include_outcome` was decoded as a `bool`,
            the equivalent `1`/`0`).
        """
        if reset not in (None, 0, 1) or include_outcome not in (True, False):
            raise RepConstructionError(
                f"({reset!r}, {include_outcome!r}) is not a valid "
                f"{type(self).__name__} (reset, include_outcome) pair"
            )
        super().__init__(qubits)
        self.reset = reset
        self.include_outcome = include_outcome


class ZBasisPrePostInstrumentRep(InstrumentRep):
    """Perfect Z-basis projection with noisy pre-/post-operations.

    For when a mid-circuit measurement can be modeled by a perfect Z-basis
    projection sandwiched by two noisy operations.
    """

    reset: Literal[None, 0, 1]
    """`None` for no reset, or `0`/`1` for reset to the corresponding state."""

    include_outcome: bool
    """Whether the measurement outcome should be recorded."""

    pre_op: GateRep
    """Noisy operation applied immediately before the projection."""

    post_op: GateRep
    """Noisy operation applied immediately after the projection."""

    _SERIALIZE_ATTRS = ["reset", "include_outcome", "pre_op", "post_op", "qubits"]

    def __init__(
        self,
        reset: Literal[None, 0, 1],
        include_outcome: bool,
        pre_op: GateRep,
        post_op: GateRep,
        qubits: str | int | Sequence[str | int] | None = (),
    ) -> None:
        """Construct a [](api:ZBasisPrePostInstrumentRep).

        Parameters
        ----------
        reset, include_outcome:
            See [](api:ZBasisProjectionInstrumentRep).

        pre_op:
            The noisy operation applied immediately before the
            projection.

        post_op:
            The noisy operation applied immediately after the projection.

        qubits:
            Qubit label(s) this operation acts upon, or `None`/unattached
            if not yet known.

        Raises
        ------
        RepConstructionError
            If `pre_op`/`post_op` aren't both [](api:GateRep) instances
            acting on `qubits`, or if `reset` is out of range.
        """
        if not isinstance(pre_op, GateRep) or not isinstance(post_op, GateRep):
            raise RepConstructionError(
                f"pre_op={pre_op!r}/post_op={post_op!r} must both be "
                "GateRep instances"
            )
        if reset not in (None, 0, 1):
            raise RepConstructionError(
                f"{reset!r} is not a valid reset value (expected None, "
                "0, or 1)"
            )
        super().__init__(qubits)
        if pre_op.qubits != self.qubits or post_op.qubits != self.qubits:
            raise RepConstructionError(
                f"pre_op.qubits={pre_op.qubits!r}/"
                f"post_op.qubits={post_op.qubits!r} must both equal "
                f"qubits={self.qubits!r}"
            )
        self.reset = reset
        self.include_outcome = include_outcome
        self.pre_op = pre_op
        self.post_op = post_op


class ZBasisOutcomeOperationDictInstrumentRep(InstrumentRep):
    """Dict with MCM outcome labels and CP map operation keys.

    For when a mid-circuit measurement can be modeled by a `pyGSTi`-like
    quantum instrument.
    """

    outcome_ops: Mapping[Hashable, GateRep]
    """Mapping from outcome label to the CP map operation for that outcome."""

    include_outcome: bool
    """Whether the measurement outcome should be recorded.

    E.g. `include_outcome=False` would look like a noisy reset.
    """

    _SERIALIZE_ATTRS = ["outcome_ops", "include_outcome", "qubits"]

    def __init__(
        self,
        outcome_ops: Mapping[Hashable, GateRep],
        include_outcome: bool,
        qubits: str | int | Sequence[str | int] | None = (),
    ) -> None:
        """Construct a [](api:ZBasisOutcomeOperationDictInstrumentRep).

        Parameters
        ----------
        outcome_ops:
            Mapping from outcome label to the CP map operation for that
            outcome.

        include_outcome:
            Whether the measurement outcome should be recorded.

        qubits:
            Qubit label(s) this operation acts upon, or `None`/unattached
            if not yet known.

        Raises
        ------
        RepConstructionError
            If `outcome_ops` isn't a `Mapping`, or any value isn't a
            [](api:GateRep) instance.
        """
        if not isinstance(outcome_ops, Mapping) or not all(
            isinstance(v, GateRep) for v in outcome_ops.values()
        ):
            raise RepConstructionError(
                f"{outcome_ops!r} is not a valid {type(self).__name__} "
                "payload (expected a Mapping with GateRep values)"
            )
        super().__init__(qubits)
        self.outcome_ops = outcome_ops
        self.include_outcome = include_outcome


class StimCircuitInstrumentRep(StimCircuitPayloadMixin, InstrumentRep):
    """STIM circuit string representation for an instrument.

    This is the same as [](api:StimCircuitGateRep), except that it should
    only be a measurement gate, i.e. one of {M, MX, MY, MZ, MR, MRX, MRY,
    MRZ, R, RX, RY, RZ, MXX, MYY, MZZ}. These are analogous to the
    following [](api:ZBasisProjectionInstrumentRep) specifications, except
    in all bases instead of just Z:

    - The first four (i.e., start with "M") are like `(None, True)`, i.e.,
      don't reset but record this outcome.
    - The second four (i.e., start with "MR") are like `(0, True)`, i.e.,
      reset to 0 and also record this outcome.
    - The third four (i.e., start with "R") are like `(0, False)`, i.e.,
      reset to 0 but don't record an outcome.
    - The last three do not correspond to a single qubit Z-basis
      projection, but could be considered equivalent to a circuit
      measuring the parity on an auxiliary qubit and then performing a
      `(0, True)` on the auxiliary.

    Qubit labels are placeholders indexing into [](api:OperationRep.qubits).

    See [](api:StimCircuitPayloadMixin) for the shared storage/construction
    logic this class shares with [](api:StimCircuitGateRep).
    """
