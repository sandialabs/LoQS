#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import copy
from typing import ClassVar

from loqs.internal import Displayable


class RepConstructionError(Exception):
    """Raised when a raw value cannot be converted into an [](api:OperationRep) subclass.

    This is raised by a concrete [](api:OperationRep) subclass's `from_raw`
    classmethod (and by [](api:upgrade_gate_rep)/[](api:upgrade_instrument_rep),
    which call it) whenever a given raw value cannot be interpreted as that
    class's expected payload shape. Catching this exception (rather than a
    bare `ValueError`) around code that tries several candidate rep classes
    in sequence ensures that a genuine bug inside a conversion routine
    propagates instead of being silently treated as "this value doesn't
    match this rep type."
    """


class OperationRep(ABC, Displayable):
    """Abstract base class for gate and instrument operation representations.

    An [](api:OperationRep) describes both the qubits an operation acts on and,
    via its concrete subclass, exactly how the operation itself is represented
    (e.g. as a unitary matrix, a STIM circuit string, a set of Kraus operators,
    etc.). The concrete subclass's identity carries the "tag" that used to be
    a separate `reptype` enum value: instead of a single untyped `rep` payload
    field paired with an enum tag, each concrete subclass exposes descriptive,
    named fields for exactly the data it needs.
    """

    qubits: tuple[str | int, ...]
    """Qubit labels that this representation should be applied to."""

    _SERIALIZE_ATTRS: ClassVar[list[str]] = ["qubits"]

    def __init__(self, qubits: str | int | Sequence[str | int] = ()) -> None:
        """Initialize the qubits this representation applies to.

        Parameters
        ----------
        qubits:
            Qubit label(s) this operation acts upon. A bare `str`/`int` is
            wrapped into a single-element tuple; any other sequence is
            converted to a tuple.
        """
        if isinstance(qubits, (str, int)):
            self.qubits = (qubits,)
        else:
            self.qubits = tuple(qubits)

    @classmethod
    @abstractmethod
    def matches(cls, raw: object) -> bool:
        """Check whether `raw` structurally matches this class's expected payload.

        `raw` is a "pre-refactor-style" value, e.g. a bare `numpy.ndarray`,
        `str`, or `Sequence`/`Mapping`, of the kind users have always been
        able to pass directly (without explicitly constructing an
        [](api:OperationRep) subclass) to e.g. [](api:DictNoiseModel).

        Parameters
        ----------
        raw:
            The raw, pre-refactor-style value to check.

        Returns
        -------
        bool
            `True` if `raw` can be converted to this class via `from_raw`.
        """
        ...

    @classmethod
    @abstractmethod
    def from_raw(
        cls, raw: object, qubits: str | int | Sequence[str | int] = (), **kwargs
    ) -> "OperationRep":
        """Construct an instance of this class from a raw, pre-refactor-style payload.

        Implementations should assume `matches(raw)` is `True`; if it is not,
        implementations should raise [](api:RepConstructionError).

        Parameters
        ----------
        raw:
            The raw, pre-refactor-style value to convert.

        qubits:
            Qubit label(s) this operation acts upon.

        **kwargs:
            Additional class-specific construction parameters. Most concrete
            classes ignore these; the composite instrument representations
            that recursively contain other [](api:GateRep) values (
            [](api:ZBasisPrePostInstrumentRep),
            [](api:ZBasisOutcomeOperationDictInstrumentRep)) require a
            `gate_upgrader` keyword argument (a callable converting a raw
            gate-level value plus qubits into a [](api:GateRep) instance)
            to recursively upgrade their nested gate-level payloads.

        Returns
        -------
        OperationRep
            The constructed instance.

        Raises
        ------
        RepConstructionError
            If `raw` cannot be interpreted as this class's payload.
        """
        ...

    def __str__(self) -> str:
        attrs = ", ".join(
            f"{attr}={getattr(self, attr)!r}" for attr in self._SERIALIZE_ATTRS
        )
        return f"{type(self).__name__}({attrs})"

    def with_qubits(self, qubits: str | int | Sequence[str | int]) -> "OperationRep":
        """Return a shallow copy of this representation retargeted onto different qubits.

        Used when a representation was looked up via a name-only key (no
        qubits attached) and needs to be attached to the qubits from the
        actual circuit label that matched, without needing to know the
        concrete subclass's specific payload field name(s).

        Parameters
        ----------
        qubits:
            The new qubit label(s) to attach.

        Returns
        -------
        OperationRep
            A shallow copy of `self` with `qubits` replaced.
        """
        new = copy.copy(self)
        if isinstance(qubits, (str, int)):
            new.qubits = (qubits,)
        else:
            new.qubits = tuple(qubits)
        return new


def is_rep_compatible(
    output_cls: type[OperationRep], accepted: Sequence[type[OperationRep]]
) -> bool:
    """Check whether `output_cls` is compatible with any class in `accepted`.

    Compatibility is `issubclass`-based rather than exact-equality-based:
    `output_cls` is compatible if it is a subclass of (or the same class
    as) any entry in `accepted`. Since `issubclass(X, X)` is always `True`,
    this is a strict generalization of comparing two concrete classes for
    equality -- a caller that (as is typical) declares only exact concrete
    classes in `accepted` sees identical behavior to an equality check, but
    a caller may also declare a coarse-grained capability like
    `accepted=[InstrumentRep]` to mean "accepts any instrument
    representation."

    Parameters
    ----------
    output_cls:
        The candidate class to check.

    accepted:
        The classes to check `output_cls` against.

    Returns
    -------
    bool
        `True` if `output_cls` is a subclass of any class in `accepted`.
    """
    return any(issubclass(output_cls, acc) for acc in accepted)
