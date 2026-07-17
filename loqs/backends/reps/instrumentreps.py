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
from types import NoneType
from typing import Callable, Literal

from loqs.backends.reps.base import (
    OperationRep,
    RepConstructionError,
    StimCircuitPayloadMixin,
)
from loqs.backends.reps.gatereps import GateRep

GateUpgrader = Callable[[object, "str | int | Sequence[str | int]"], GateRep]
"""A callable converting a raw gate-level payload and qubits into a [](api:GateRep).

Passed as the `gate_upgrader` keyword argument to `from_raw` for the
composite instrument representations ([](api:ZBasisPrePostInstrumentRep),
[](api:ZBasisOutcomeOperationDictInstrumentRep)) that recursively contain
gate-level payloads, since a generic `from_raw` classmethod has no way to
know which concrete [](api:GateRep) classes a given caller considers valid.
"""


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
        qubits: str | int | Sequence[str | int] = (),
    ) -> None:
        super().__init__(qubits)
        self.reset = reset
        self.include_outcome = include_outcome

    @classmethod
    def matches(cls, raw: object) -> bool:
        """Check that `raw` is a `(reset, include_outcome)` 2-tuple/list."""
        if not isinstance(raw, (tuple, list)):
            return False
        if len(raw) != 2:
            return False
        if not isinstance(raw[0], (int, NoneType)):
            return False
        if not isinstance(raw[1], bool):
            return False
        return True

    @classmethod
    def from_raw(
        cls, raw: object, qubits: str | int | Sequence[str | int] = (), **kwargs
    ) -> "ZBasisProjectionInstrumentRep":
        if not cls.matches(raw):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a "
                "(reset, include_outcome) pair)"
            )
        assert isinstance(raw, (tuple, list))
        return cls(raw[0], raw[1], qubits)


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
        qubits: str | int | Sequence[str | int] = (),
    ) -> None:
        super().__init__(qubits)
        self.reset = reset
        self.include_outcome = include_outcome
        self.pre_op = pre_op
        self.post_op = post_op

    @classmethod
    def matches(cls, raw: object) -> bool:
        """Check that `raw` is a `(pre_op_raw, post_op_raw)` 2-tuple/list.

        Note that this structurally overlaps with
        [](api:ZBasisProjectionInstrumentRep)'s expected shape (both are
        length-2 tuples/lists). Callers using `matches`-based dispatch
        (e.g. [](api:upgrade_instrument_rep)) must check
        [](api:ZBasisProjectionInstrumentRep) first, mirroring the priority
        order of the `if`/`elif` chain this replaces.
        """
        return isinstance(raw, (tuple, list)) and len(raw) == 2

    @classmethod
    def from_raw(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] = (),
        reset: Literal[None, 0, 1] = None,
        include_outcome: bool = True,
        gate_upgrader: GateUpgrader | None = None,
        **kwargs,
    ) -> "ZBasisPrePostInstrumentRep":
        """Construct a [](api:ZBasisPrePostInstrumentRep) from a raw payload.

        Parameters
        ----------
        raw:
            A `(pre_op_raw, post_op_raw)` pair of raw, pre-refactor-style
            gate-level payloads.

        qubits:
            Qubit label(s) this operation acts upon.

        reset, include_outcome:
            Passed through directly to the constructor; unlike `pre_op`/
            `post_op`, these are not derived from `raw` itself.

        gate_upgrader:
            Required. A callable used to recursively convert `raw[0]`/
            `raw[1]` into [](api:GateRep) instances.
        """
        if not cls.matches(raw):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a "
                "(pre_op, post_op) pair)"
            )
        if gate_upgrader is None:
            raise RepConstructionError(
                f"{cls.__name__}.from_raw requires a `gate_upgrader` callable "
                "to convert the raw pre-/post-operation payloads"
            )
        assert isinstance(raw, (tuple, list))
        pre_op = gate_upgrader(raw[0], qubits)
        post_op = gate_upgrader(raw[1], qubits)
        return cls(reset, include_outcome, pre_op, post_op, qubits)


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
        qubits: str | int | Sequence[str | int] = (),
    ) -> None:
        super().__init__(qubits)
        self.outcome_ops = outcome_ops
        self.include_outcome = include_outcome

    @classmethod
    def matches(cls, raw: object) -> bool:
        return isinstance(raw, Mapping)

    @classmethod
    def from_raw(
        cls,
        raw: object,
        qubits: str | int | Sequence[str | int] = (),
        include_outcome: bool = True,
        gate_upgrader: GateUpgrader | None = None,
        **kwargs,
    ) -> "ZBasisOutcomeOperationDictInstrumentRep":
        """Construct a [](api:ZBasisOutcomeOperationDictInstrumentRep) from a raw payload.

        Parameters
        ----------
        raw:
            A mapping from outcome label to a raw, pre-refactor-style
            gate-level payload.

        qubits:
            Qubit label(s) this operation acts upon.

        include_outcome:
            Passed through directly to the constructor.

        gate_upgrader:
            Required. A callable used to recursively convert each value of
            `raw` into a [](api:GateRep) instance.
        """
        if not cls.matches(raw):
            raise RepConstructionError(
                f"{raw!r} is not a valid {cls.__name__} payload (expected a Mapping)"
            )
        if gate_upgrader is None:
            raise RepConstructionError(
                f"{cls.__name__}.from_raw requires a `gate_upgrader` callable "
                "to convert the raw outcome-operation payloads"
            )
        assert isinstance(raw, Mapping)
        outcome_ops = {k: gate_upgrader(v, qubits) for k, v in raw.items()}
        return cls(outcome_ops, include_outcome, qubits)


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
