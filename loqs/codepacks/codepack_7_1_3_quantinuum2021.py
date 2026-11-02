#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Deprecated function wrappers for the [[7,1,3]] color code codepack.

Use [](api:SteaneCodepack) instead, which takes these functions' options
as constructor arguments and builds the code, its ideal model and its
transversal CX from one object.
"""

from collections.abc import Sequence

from loqs.backends.circuit.basecircuit import BasePhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.backends.model.basemodel import BaseNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    QSimSuperopGateRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.codepacks.steane import SteaneCodepack
from loqs.internal.legacy import deprecated


@deprecated("SteaneCodepack(...).create_qec_code()")
def create_qec_code(
    ft_state_prep_max_repeats: int = 100,
    include_idles: bool = False,
    gate_durations: dict[str, int | float] | None = None,
    idle_gates: dict[int | float, str] | None = None,
    circuit_backend: type[BasePhysicalCircuit] = PyGSTiPhysicalCircuit,
):
    """Create a [](api:QECCode) implementing the [[7,1,3]] code.

    The parameters are the options of [](api:SteaneCodepack), which this
    returns `create_qec_code()` of.
    """
    return SteaneCodepack(
        ft_state_prep_max_repeats=ft_state_prep_max_repeats,
        include_idles=include_idles,
        gate_durations=gate_durations,  # type: ignore[arg-type]
        idle_gates=idle_gates,  # type: ignore[arg-type]
        circuit_backend=circuit_backend,
    ).create_qec_code()


@deprecated("SteaneCodepack(...).create_ideal_model(...)")
def create_ideal_model(
    qubits: Sequence[str],
    model_backend: type[BaseNoiseModel] = PyGSTiNoiseModel,
    gaterep: type[GateRep] = QSimSuperopGateRep,
    instrep: type[InstrumentRep] = ZBasisProjectionInstrumentRep,
):
    """Create an ideal (noiseless) model for the [[7,1,3]] code.

    The model holds the gates of [](api:SteaneCodepack)'s code with idles
    included, as before.
    """
    return SteaneCodepack(
        include_idles=True, circuit_backend=ListPhysicalCircuit
    ).create_ideal_model(
        qubits,
        model_backend=model_backend,
        gaterep=gaterep,
        instrep=instrep,
    )
