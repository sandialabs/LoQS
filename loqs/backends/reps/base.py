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
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from loqs.internal import Displayable
from loqs.internal.serializable import MisformedDecodableError


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

    qubit_labels: tuple[str | int, ...]
    """Qubit labels that this representation should be applied to."""

    _SERIALIZE_ATTRS: ClassVar[list[str]] = ["qubit_labels"]

    _SERIALIZE_ATTRS_MAP: ClassVar[dict[str, str]] = {"qubits": "qubit_labels"}
    """Lets old serialized files (which stored this field under the
    pre-rename key `"qubits"`) still decode correctly: `_from_decoded_attrs`
    (see [](api:Serializable._from_decoded_attrs)) maps a decoded
    `"qubits"` key to the modern `qubit_labels` constructor kwarg."""

    @abstractmethod
    def __init__(
        self, qubit_labels: str | int | Sequence[str | int] | None = ()
    ) -> None:
        """Initialize the qubits this representation applies to.

        Marked abstract only to keep this class and [](api:GateRep)/
        [](api:InstrumentRep) non-instantiable; every concrete leaf class
        overrides `__init__` and so satisfies Python's `abc` machinery.

        Parameters
        ----------
        qubit_labels:
            Qubit label(s) this operation acts upon. A bare `str`/`int` is
            wrapped into a single-element tuple; any other sequence is
            converted to a tuple; `None` is treated as the empty tuple
            (no qubits attached yet -- see
            [](api:OperationRep.with_qubit_labels)).
        """
        if qubit_labels is None:
            self.qubit_labels = ()
        elif isinstance(qubit_labels, (str, int)):
            self.qubit_labels = (qubit_labels,)
        else:
            self.qubit_labels = tuple(qubit_labels)

    def __str__(self) -> str:
        attrs = ", ".join(
            f"{attr}={getattr(self, attr)!r}" for attr in self._SERIALIZE_ATTRS
        )
        return f"{type(self).__name__}({attrs})"

    @classmethod
    def _from_decoded_attrs(cls, attr_dict: Mapping[str, Any]) -> "OperationRep":
        # `OperationRep` is only ever the *recorded* class for a value
        # serialized under the pre-refactor RepTuple(rep, qubits, reptype)
        # format (only concrete leaf classes are instantiable, so current
        # code never produces a serialized object whose recorded class is
        # this abstract base) -- issue #97's complete removal of RepTuple
        # redirects its old (module, class) straight here via
        # IMPORT_LOCATION_CHANGES_BY_VERSION, so this absorbs exactly the
        # dispatch RepTuple._from_decoded_attrs used to do. Deferred
        # import avoids a circular dependency, since legacy.py itself
        # imports from this module (via gatereps.py/instrumentreps.py).
        if cls is OperationRep:
            from loqs.backends.reps.legacy import (
                _LegacyGateRepValue,
                _LegacyInstrumentRepValue,
                _upgrade_legacy_gaterep,
                _upgrade_legacy_instrumentrep,
            )

            rep = attr_dict["rep"]
            qubits = attr_dict["qubits"]
            reptype = attr_dict["reptype"]
            if isinstance(reptype, _LegacyGateRepValue):
                return _upgrade_legacy_gaterep(reptype, rep, qubits)
            elif isinstance(reptype, _LegacyInstrumentRepValue):
                return _upgrade_legacy_instrumentrep(reptype, rep, qubits)
            raise MisformedDecodableError(
                f"Unrecognized legacy RepTuple reptype {reptype!r}"
            )
        return super()._from_decoded_attrs(attr_dict)

    def with_qubit_labels(
        self, qubit_labels: str | int | Sequence[str | int]
    ) -> "OperationRep":
        """Return a copy of this representation retargeted onto different qubits.

        Reconstructs via `type(self)`'s own `__init__` (also retargeting
        any nested `OperationRep`/`Mapping` fields), so the result is
        re-validated and can't drift from its own type contract.

        Parameters
        ----------
        qubit_labels:
            The new qubit label(s) to attach.

        Returns
        -------
        OperationRep
            A new `type(self)` instance with `qubit_labels` replaced.

        Raises
        ------
        RepConstructionError
            If retargeting the payload onto `qubit_labels` would be invalid.
        """
        kwargs = {}
        for attr in self._SERIALIZE_ATTRS:
            if attr == "qubit_labels":
                kwargs[attr] = qubit_labels
                continue
            value = getattr(self, attr)
            if isinstance(value, OperationRep):
                value = value.with_qubit_labels(qubit_labels)
            elif isinstance(value, Mapping):
                value = {
                    k: (
                        v.with_qubit_labels(qubit_labels)
                        if isinstance(v, OperationRep)
                        else v
                    )
                    for k, v in value.items()
                }
            kwargs[attr] = value
        return type(self)(**kwargs)


class StimCircuitPayloadMixin:
    """Shared payload/construction/validation for [](api:StimCircuitGateRep)
    and [](api:StimCircuitInstrumentRep), which both wrap a STIM
    circuit-string template with placeholder qubit indices. Also used as
    a shared `isinstance` type for callers that need to treat either
    uniformly (e.g. [](api:DictNoiseModel)'s STIM-text merging).
    """

    circuit_str: str
    """The STIM circuit-string template, with placeholder qubit indices."""

    _SERIALIZE_ATTRS: ClassVar[list[str]] = ["circuit_str", "qubit_labels"]

    def __init__(
        self,
        circuit_str: str,
        qubit_labels: str | int | Sequence[str | int] | None = (),
    ) -> None:
        if not isinstance(circuit_str, str):
            raise RepConstructionError(
                f"{circuit_str!r} is not a valid {type(self).__name__} "
                "payload (expected a str)"
            )
        super().__init__(qubit_labels)  # type: ignore[call-arg]
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
