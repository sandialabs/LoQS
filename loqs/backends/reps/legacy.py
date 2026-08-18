#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from loqs.backends.reps.base import OperationRep
from loqs.backends.reps.gatereps import (
    GateRep,
    KrausGateRep,
    PTMGateRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.backends.reps.instrumentreps import (
    InstrumentRep,
    StimCircuitInstrumentRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.internal.serializable import MisformedDecodableError


class _LegacyGateRepValue(Enum):
    """Decode-only mirror of the pre-refactor `GateRep` enum.

    This exists purely to tag values decoded from old serialized files (see
    `GateRep._from_decoded_attrs`); it is never constructed by new code.
    Names and values are a byte-for-byte copy of the original `GateRep`
    enum so that old integer `value` fields resolve to the same member.
    """

    UNITARY = 1
    PTM = 2
    QSIM_SUPEROPERATOR = 3
    STIM_CIRCUIT_STR = 4
    PROBABILISTIC_STIM_OPERATIONS = 5
    KRAUS_OPERATORS = 6


class _LegacyInstrumentRepValue(Enum):
    """Decode-only mirror of the pre-refactor `InstrumentRep` enum.

    See `_LegacyGateRepValue`. A separate enum (rather than reusing
    `_LegacyGateRepValue`) is required because
    `InstrumentRep.STIM_CIRCUIT_STR` and `GateRep.STIM_CIRCUIT_STR`
    coincidentally shared the same integer value (4) in the pre-refactor
    enums; keeping them as two distinct Python types preserves that
    provenance instead of conflating the two.
    """

    ZBASIS_PROJECTION = 1
    ZBASIS_PRE_POST_OPERATIONS = 2
    ZBASIS_OUTCOME_OPERATION_DICT = 3
    STIM_CIRCUIT_STR = 4


_LEGACY_GATEREP_CLASS: dict[_LegacyGateRepValue, type[GateRep]] = {
    _LegacyGateRepValue.UNITARY: UnitaryGateRep,
    _LegacyGateRepValue.PTM: PTMGateRep,
    _LegacyGateRepValue.QSIM_SUPEROPERATOR: QSimSuperopGateRep,
    _LegacyGateRepValue.STIM_CIRCUIT_STR: StimCircuitGateRep,
    _LegacyGateRepValue.PROBABILISTIC_STIM_OPERATIONS: ProbabilisticStimGateRep,
    _LegacyGateRepValue.KRAUS_OPERATORS: KrausGateRep,
}
"""Maps each legacy `GateRep` enum-member tag to its modern concrete class."""

_LEGACY_INSTRUMENTREP_CLASS: dict[_LegacyInstrumentRepValue, type[InstrumentRep]] = {
    _LegacyInstrumentRepValue.ZBASIS_PROJECTION: ZBasisProjectionInstrumentRep,
    _LegacyInstrumentRepValue.ZBASIS_PRE_POST_OPERATIONS: ZBasisPrePostInstrumentRep,
    _LegacyInstrumentRepValue.ZBASIS_OUTCOME_OPERATION_DICT: (
        OutcomeOperationDictInstrumentRep
    ),
    _LegacyInstrumentRepValue.STIM_CIRCUIT_STR: StimCircuitInstrumentRep,
}
"""Maps each legacy `InstrumentRep` enum-member tag to its modern concrete class."""


def _upgrade_legacy_tag(
    value: object,
    primary_enum: type[Enum],
    primary_class_map: Mapping[Enum, type[OperationRep]],
    secondary_enum: type[Enum],
    secondary_class_map: Mapping[Enum, type[OperationRep]],
) -> object:
    """Shared dispatch for `upgrade_legacy_gaterep_tag`/`upgrade_legacy_instrumentrep_tag`.

    Both functions do the exact same three-way dispatch (their own enum
    family, the other enum family, or a bare `int` assumed to belong to
    their own family) -- only which family is "primary" differs, so that's
    the only thing parameterized here.
    """
    if isinstance(value, primary_enum):
        return primary_class_map[value]
    elif isinstance(value, secondary_enum):
        return secondary_class_map[value]
    elif isinstance(value, int):
        return primary_class_map[primary_enum(value)]
    return value


def upgrade_legacy_gaterep_tag(value: object) -> object:
    """Translate a decoded `_gatereps`-style tag to its modern form.

    New-style files serialize a model's `_gatereps`/`_instreps` entries as
    bare `type[GateRep]`/`type[InstrumentRep]` classes, which decode
    directly to the correct class with no translation needed (classes are
    natively encodable/decodable via `Serializable`). Files serialized at
    `SERIALIZATION_VERSION` 1 (the format introduced alongside HDF5
    support) instead serialized each entry as a `GateRep`/`InstrumentRep`
    *enum member*, which decodes (via `GateRep._from_decoded_attrs`/
    `InstrumentRep._from_decoded_attrs`) to a `_LegacyGateRepValue`/
    `_LegacyInstrumentRepValue` tag rather than a class; this function maps
    that tag to the corresponding modern concrete class.

    Files serialized at `SERIALIZATION_VERSION` 0 (the original,
    JSON-only, pre-HDF5 format) go a step further: `_gatereps`/`_instreps`
    entries were encoded as bare raw `int`s with no `(module, class)`
    metadata at all -- the pre-refactor code relied on the caller already
    knowing which enum to construct (`GateRep(v)` for `_gatereps` vs.
    `InstrumentRep(v)` for `_instreps`), since a bare int carries no
    information about which of the two enums it came from. This function
    is only ever called on `_gatereps` entries, so a bare `int` is assumed
    to be a legacy `GateRep` value; see `upgrade_legacy_instrumentrep_tag`
    for the `_instreps`-context equivalent.

    Any other value (e.g. an already-resolved class) is passed through
    unchanged.
    """
    return _upgrade_legacy_tag(
        value,
        _LegacyGateRepValue,
        _LEGACY_GATEREP_CLASS,
        _LegacyInstrumentRepValue,
        _LEGACY_INSTRUMENTREP_CLASS,
    )


def upgrade_legacy_instrumentrep_tag(value: object) -> object:
    """Translate a decoded `_instreps`-style tag to its modern form.

    Identical to `upgrade_legacy_gaterep_tag`, except a bare `int` (from a
    `SERIALIZATION_VERSION` 0 file) is assumed to be a legacy
    `InstrumentRep` value instead of a `GateRep` one, since this is only
    ever called on `_instreps` entries.
    """
    return _upgrade_legacy_tag(
        value,
        _LegacyInstrumentRepValue,
        _LEGACY_INSTRUMENTREP_CLASS,
        _LegacyGateRepValue,
        _LEGACY_GATEREP_CLASS,
    )


def _upgrade_legacy_gaterep(
    legacy_value: _LegacyGateRepValue, rep: Any, qubits: tuple
) -> GateRep:
    """Reshape an old `(rep, qubits, reptype)` gate payload into a new `GateRep`."""
    if legacy_value is _LegacyGateRepValue.UNITARY:
        return UnitaryGateRep(unitary=rep, qubit_labels=qubits)
    elif legacy_value is _LegacyGateRepValue.PTM:
        return PTMGateRep(ptm=rep, qubit_labels=qubits)
    elif legacy_value is _LegacyGateRepValue.QSIM_SUPEROPERATOR:
        return QSimSuperopGateRep(superop=rep, qubit_labels=qubits)
    elif legacy_value is _LegacyGateRepValue.STIM_CIRCUIT_STR:
        return StimCircuitGateRep(circuit_str=rep, qubit_labels=qubits)
    elif legacy_value is _LegacyGateRepValue.PROBABILISTIC_STIM_OPERATIONS:
        return ProbabilisticStimGateRep(operations=rep, qubit_labels=qubits)
    elif legacy_value is _LegacyGateRepValue.KRAUS_OPERATORS:
        # Skip the TP check: this decodes already-accepted data, not new input.
        return KrausGateRep(
            kraus_operators=rep, qubit_labels=qubits, tp_check_abstol=None
        )
    raise MisformedDecodableError(
        f"Unrecognized legacy GateRep value {legacy_value!r}"
    )


def _upgrade_legacy_instrumentrep(
    legacy_value: _LegacyInstrumentRepValue, rep: Any, qubits: tuple
) -> InstrumentRep:
    """Reshape an old `(rep, qubits, reptype)` instrument payload into a new `InstrumentRep`.

    For `ZBASIS_PRE_POST_OPERATIONS`/`ZBASIS_OUTCOME_OPERATION_DICT`, the
    nested gate-level entries of `rep` (originally themselves `RepTuple`
    objects) have already been upgraded to concrete `GateRep` instances by
    the time this function runs: attribute decoding happens bottom-up,
    before the outer `RepTuple`'s own `_from_decoded_attrs` is invoked, so
    no recursive re-upgrading is needed here.
    """
    if legacy_value is _LegacyInstrumentRepValue.ZBASIS_PROJECTION:
        reset, include_outcome = rep
        return ZBasisProjectionInstrumentRep(
            reset=reset, include_outcome=include_outcome, qubit_labels=qubits
        )
    elif legacy_value is _LegacyInstrumentRepValue.ZBASIS_PRE_POST_OPERATIONS:
        reset, include_outcome, pre_op, post_op = rep
        return ZBasisPrePostInstrumentRep(
            reset=reset,
            include_outcome=include_outcome,
            pre_op=pre_op,
            post_op=post_op,
            qubit_labels=qubits,
        )
    elif legacy_value is _LegacyInstrumentRepValue.ZBASIS_OUTCOME_OPERATION_DICT:
        outcome_ops, include_outcome = rep
        return OutcomeOperationDictInstrumentRep(
            outcome_ops=outcome_ops,
            include_outcome=include_outcome,
            qubit_labels=qubits,
        )
    elif legacy_value is _LegacyInstrumentRepValue.STIM_CIRCUIT_STR:
        return StimCircuitInstrumentRep(circuit_str=rep, qubit_labels=qubits)
    raise MisformedDecodableError(
        f"Unrecognized legacy InstrumentRep value {legacy_value!r}"
    )
