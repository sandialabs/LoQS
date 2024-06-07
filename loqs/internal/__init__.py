"""Utility classes and functions for LoQS.
"""

from typing import TypeAlias


# Convenience type aliases
# These are just to signal to users that these ints/str
# are meant to be more restricted (although we don't force it here)
Bit: TypeAlias = int
PauliStr: TypeAlias = str
