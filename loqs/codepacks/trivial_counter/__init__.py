#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################
"""A trivial `LoQS` QEC codepack for testing purposes.
This implementation provides a simple counter-like behavior where a "state"
value can be incremented. This is purely for demonstration and testing purposes.
"""

import time

from loqs.codepacks.base import BaseCodepack
from loqs.core import Instruction, QECCode
from loqs.core.frame import Frame


class TrivialCounterCodepack(BaseCodepack):
    """A classical counter with increment, initialize and sleep instructions.

    It has no configuration. Its code uses no gates, so its ideal model
    is empty; pass `model_backend=DictNoiseModel` to
    `create_ideal_model()` to avoid needing pyGSTi.
    """

    def create_qec_code(self) -> QECCode:
        """Create a trivial QECCode for testing.

        Returns
        -------
        QECCode
            A [](api:QECCode) implementing the trivial counter code.
        """
        # A single placeholder "qubit" gives a minimal template structure
        qubits = ["Q0"]
        data_qubits = ["Q0"]
        instructions: dict[str, Instruction] = {}

        def increment_apply_fn(counter: int, increment_by: int) -> Frame:
            """Apply function for the increment instruction."""
            return Frame({"counter": counter + increment_by})

        def init_counter_apply_fn(initial_value: int) -> Frame:
            """Apply function to initialize the state."""
            return Frame({"counter": initial_value})

        def sleep_apply_fn(duration: float, counter: int | None = None):
            """Apply function for the sleep instruction."""
            time.sleep(duration)
            return Frame({"counter": counter})

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

        instructions["Sleep"] = Instruction(
            sleep_apply_fn,
            data={"duration": 0.0, "counter": None},
            param_priorities={"counter": ["history[-1]", "instruction"]},
            name="Sleep for duration",
        )

        return QECCode(
            instructions, qubits, data_qubits, "Trivial Counter Code"
        )
