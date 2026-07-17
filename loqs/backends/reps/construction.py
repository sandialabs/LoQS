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

from loqs.backends.reps.base import RepConstructionError
from loqs.backends.reps.gatereps import GateRep
from loqs.backends.reps.instrumentreps import InstrumentRep


def upgrade_gate_rep(
    raw: object,
    qubits: str | int | Sequence[str | int],
    allowed: Sequence[type[GateRep]],
    **kwargs,
) -> GateRep:
    """Upgrade a raw, pre-refactor-style gate payload to a [](api:GateRep).

    If `raw` is already a [](api:GateRep) instance, it is returned as-is
    (after confirming its type is one of `allowed`). Otherwise, each class
    in `allowed` is tried in order via its `matches` classmethod, and the
    first match is used to construct the result via `from_raw`.

    Parameters
    ----------
    raw:
        The raw gate payload, or an already-constructed [](api:GateRep).

    qubits:
        Qubit label(s) this operation acts upon.

    allowed:
        The candidate [](api:GateRep) classes to try, in priority order.
        Order matters: some classes' `matches` checks structurally overlap
        (see e.g. [](api:UnitaryGateRep.matches)), so the first class in
        `allowed` whose `matches` returns `True` wins.

    **kwargs:
        Forwarded to the matched class's `from_raw`.

    Returns
    -------
    GateRep
        The upgraded (or passed-through) representation.

    Raises
    ------
    RepConstructionError
        If `raw` cannot be matched to any class in `allowed`.
    """
    if isinstance(raw, GateRep):
        assert any(
            isinstance(raw, cls) for cls in allowed
        ), f"Provided {raw} but its type is not in the allowed reps {allowed}"
        return raw
    for cls in allowed:
        if cls.matches(raw):
            return cls.from_raw(raw, qubits, **kwargs)
    raise RepConstructionError(
        f"Could not match {raw!r} to any of {list(allowed)}"
    )


def upgrade_instrument_rep(
    raw: object,
    qubits: str | int | Sequence[str | int],
    allowed: Sequence[type[InstrumentRep]],
    **kwargs,
) -> InstrumentRep:
    """Upgrade a raw, pre-refactor-style instrument payload to an [](api:InstrumentRep).

    See [](api:upgrade_gate_rep) for the general behavior; this is the
    identical mechanism applied to [](api:InstrumentRep) subclasses. The
    two composite instrument representations
    ([](api:ZBasisPrePostInstrumentRep),
    [](api:ZBasisOutcomeOperationDictInstrumentRep)) require a
    `gate_upgrader` keyword argument to be supplied via `**kwargs` (see
    their `from_raw` documentation).

    Raises
    ------
    RepConstructionError
        If `raw` cannot be matched to any class in `allowed`.
    """
    if isinstance(raw, InstrumentRep):
        assert any(
            isinstance(raw, cls) for cls in allowed
        ), f"Provided {raw} but its type is not in the allowed reps {allowed}"
        return raw
    for cls in allowed:
        if cls.matches(raw):
            return cls.from_raw(raw, qubits, **kwargs)
    raise RepConstructionError(
        f"Could not match {raw!r} to any of {list(allowed)}"
    )
