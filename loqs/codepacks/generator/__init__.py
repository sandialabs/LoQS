#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""The codepack generator: specs as pure data, and what's derived from them.

A codepack's spec (`loqs.codepacks.generator.schema`) holds its facts,
circuits and declared logical gates. The generator derives the frame and
syndrome maps from the declared logical action with Stim.
"""

from loqs.codepacks.generator.codepack import (
    GeneratedCodepack,
    StaleArtifactError,
)
from loqs.codepacks.generator.schema import (
    CheckCircuit,
    Circuit,
    CodeSpec,
    Composite,
    CustomApplyFn,
    Facts,
    Gate,
    LookupTable,
    Parallel,
    RepeatUntilSuccess,
    Role,
    SpecError,
    Src,
    Tracking,
    TwoPatchGate,
    ours,
    paper,
)

__all__ = [
    "CheckCircuit",
    "Circuit",
    "CodeSpec",
    "Composite",
    "CustomApplyFn",
    "Facts",
    "Gate",
    "GeneratedCodepack",
    "LookupTable",
    "Parallel",
    "RepeatUntilSuccess",
    "Role",
    "SpecError",
    "Src",
    "StaleArtifactError",
    "Tracking",
    "TwoPatchGate",
    "ours",
    "paper",
]
