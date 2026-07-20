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
    """Raised when a value doesn't match an [](api:OperationRep) subclass's
    expected shape or type, by that class's own `__init__` or by
    [](api:convert)."""


class OperationRep(ABC, Displayable):
    """Abstract base class for gate and instrument operation representations.

    Describes the qubits an operation acts on; each concrete subclass
    adds descriptive, named fields for exactly the data it needs, and
    validates them itself in `__init__`, raising
    [](api:RepConstructionError) for an invalid value.
    """

    qubits: tuple[str | int, ...]
    """Qubit labels that this representation should be applied to."""

    _SERIALIZE_ATTRS: ClassVar[list[str]] = ["qubits"]

    @abstractmethod
    def __init__(
        self, qubits: str | int | Sequence[str | int] | None = ()
    ) -> None:
        """Initialize the qubits this representation applies to.

        Marked abstract only to keep this class and [](api:GateRep)/
        [](api:InstrumentRep) non-instantiable; every concrete leaf class
        overrides `__init__` and so satisfies Python's `abc` machinery.

        Parameters
        ----------
        qubits:
            Qubit label(s) this operation acts upon. A bare `str`/`int` is
            wrapped into a single-element tuple; any other sequence is
            converted to a tuple; `None` is treated as the empty tuple
            (no qubits attached yet -- see [](api:OperationRep.with_qubits)).
        """
        if qubits is None:
            self.qubits = ()
        elif isinstance(qubits, (str, int)):
            self.qubits = (qubits,)
        else:
            self.qubits = tuple(qubits)

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


class StimCircuitPayloadMixin:
    """Shared payload/construction/validation for [](api:StimCircuitGateRep)
    and [](api:StimCircuitInstrumentRep), which both wrap a STIM
    circuit-string template with placeholder qubit indices. Also used as
    a shared `isinstance` type for callers that need to treat either
    uniformly (e.g. [](api:DictNoiseModel)'s STIM-text merging).
    """

    circuit_str: str
    """The STIM circuit-string template, with placeholder qubit indices."""

    _SERIALIZE_ATTRS: ClassVar[list[str]] = ["circuit_str", "qubits"]

    def __init__(
        self,
        circuit_str: str,
        qubits: str | int | Sequence[str | int] | None = (),
    ) -> None:
        if not isinstance(circuit_str, str):
            raise RepConstructionError(
                f"{circuit_str!r} is not a valid {type(self).__name__} "
                "payload (expected a str)"
            )
        super().__init__(qubits)  # type: ignore[call-arg]
        self.circuit_str = circuit_str


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
