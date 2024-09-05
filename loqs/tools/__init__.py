"""Utility classes and functions for LoQS.
"""

from typing import Literal, TypeAlias

from .castable import Castable

# Convenience type aliases
# These are just to signal to users that these ints/str
# are meant to be more restricted (although we don't force it here)
Bit: TypeAlias = Literal[0] | Literal[1]
PauliStr: TypeAlias = str
