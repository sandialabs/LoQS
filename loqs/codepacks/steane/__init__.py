#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""The Quantinuum 2021 [[7,1,3]] Steane codepack, generated from its spec."""

from loqs.codepacks.generator.codepack import GeneratedCodepack
from loqs.codepacks.steane.spec import SPEC


class SteaneCodepack(GeneratedCodepack):
    """The [[7,1,3]] Steane code of Quantinuum's 2021 fault-tolerant
    experiment (Ryan-Anderson et al., arXiv:2107.07505), on its 10-qubit
    layout: seven data qubits and three auxiliary qubits.

    See [](api:GeneratedCodepack) for the options and the lock.
    """

    spec = SPEC
    artifact = "loqs.codepacks.steane._generated"
