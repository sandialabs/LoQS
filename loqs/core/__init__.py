#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Core objects for LoQS

These are primarily objects used for high-level objects that store or
orchestrate the execution of logical qubit simulation.
"""

# First for import reasons
from .frame import Frame
from .history import History

from .instructions import (
    Instruction,
    InstructionLabel,
    InstructionStack,
    PatchGeometry,
)

# Before QECCodePatch
from .qeccode import QECCode

# After Instruction
from .recordables import (
    MeasurementOutcomes,
    QECCodePatch,
    PauliFrame,
    PatchDict,
    PatchLayout,
    PatchRelation,
)
from .syndromelabel import SyndromeLabel

# After QECCodePatch
from .quantumprogram import QuantumProgram
from .programresults import ProgramResults

from loqs.internal.legacy import install_legacy_module_aliases_for_relocations
from loqs.internal.serializable import Serializable

# Keeps every historical import path that's a pure module relocation (the
# class itself unchanged, just moved files -- e.g. SyndromeLabel's old
# loqs.core.syndrome path) importable, without needing its own explicit
# shim. A real rename (the class's own name also changed, e.g. PatchDict
# -> PatchLayout) is never auto-forwarded this way -- see
# install_legacy_module_aliases_for_relocations's own docstring for why.
install_legacy_module_aliases_for_relocations(
    Serializable._get_cumulative_changes(0)
)
