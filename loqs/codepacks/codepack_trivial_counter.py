#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################
"""A trivial LoQS QEC codepack for testing purposes.
This implementation provides a simple counter-like behavior where a "state"
value can be incremented. This is purely for demonstration and testing purposes.
"""
from collections.abc import Sequence
from typing import Mapping
from loqs.backends.model.basemodel import (
    BaseNoiseModel,
    GateRep,
    InstrumentRep,
)
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame
from loqs.core.instructions.instruction import DEFAULT_PRIORITIES


def create_qec_code():
    """Create a trivial QECCode for testing.
    This codepack implements a simple counter with an increment instruction.
    Returns
    -------
        A :class:`.QECCode` implementing the trivial counter code.
    """
    # We don't need actual qubits for this trivial example
    # but we'll define a minimal template structure
    qubits = ["Q0"]  # Single "qubit" for simplicity
    data_qubits = ["Q0"]
    instructions: dict[str, Instruction] = {}

    # Define the increment instruction
    def increment_apply_fn(counter: int, increment_by: int) -> Frame:
        """Apply function for the increment instruction."""
        new_state = counter + increment_by
        return Frame({"counter": new_state})

    # Define an instruction to initialize the state
    def init_counter_apply_fn(initial_value: int) -> Frame:
        """Apply function to initialize the state."""
        return Frame({"counter": initial_value})

    instructions["Increment"] = Instruction(
        increment_apply_fn,
        data={"increment_by": 1},
        param_priorities={"counter": ["history[-1]"]},
        name="Increment counter by 1",
    )

    instructions["Init Counter"] = Instruction(
        init_counter_apply_fn,
        data={"initial_value": 0},
        name="Initialize counter",
    )

    code = QECCode(instructions, qubits, data_qubits, "Trivial Counter Code")
    return code


def create_ideal_model(
    qubits: Sequence[str],
    model_backend: type[BaseNoiseModel] = DictNoiseModel,
    gaterep: GateRep = GateRep.QSIM_SUPEROPERATOR,
    instrep: InstrumentRep = InstrumentRep.ZBASIS_PROJECTION,
):
    """Create an ideal (noiseless) model for the trivial code.
    Since this is a trivial classical counter, we return an empty model.
    Parameters
    ----------
    qubits:
        List of qubit labels to use (not actually used in this trivial case).
    model_backend:
        The model backend to use (not actually used in this trivial case).
    Returns
    -------
        An empty dict representing no operations needed for this trivial code.
    """
    # For this trivial classical counter, we don't need any quantum operations
    # Return an empty DictNoiseModel
    return DictNoiseModel(({}, {}))
