"""Common container-like objects stored in a :class:`.Frame`.

While a :class:`.Frame` -- and by extension, a :class:`.History` --
contains many LoQS core objects or simple built-in Python objects,
there are several other kinds of objects that we commonly want
to store also.

This module serves as a place to store a lot of these objects.
"""

from .measurementoutcomes import MeasurementOutcomes
from .patchdict import PatchDict
