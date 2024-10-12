""":class:`.Instruction`-related objects and functions.

In addition to the :class:`.Instruction` itself,
this includes :class:`.InstructionLabel` and
:class`.InstructionStack` objects.

This module also include the :mod:`.builders`,
which contain functions that generate common
:class:`.Instruction` types.
"""

from .instruction import Instruction
from .instructionlabel import InstructionLabel
from .instructionstack import InstructionStack
