#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################
"""Deprecated function wrappers for the trivial counter codepack.

Use [](api:TrivialCounterCodepack) instead.
"""

from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.codepacks.trivial_counter import TrivialCounterCodepack
from loqs.internal.legacy import deprecated


@deprecated("TrivialCounterCodepack().create_qec_code()")
def create_qec_code():
    """Create the trivial counter [](api:QECCode)."""
    return TrivialCounterCodepack().create_qec_code()


@deprecated("TrivialCounterCodepack().create_ideal_model()")
def create_ideal_model(qubits=None):
    """Create an empty [](api:DictNoiseModel) for the trivial counter."""
    return TrivialCounterCodepack().create_ideal_model(
        qubits or [], model_backend=DictNoiseModel
    )
