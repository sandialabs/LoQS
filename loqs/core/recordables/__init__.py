#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Common container-like objects stored in a :class:`.Frame`.

While a :class:`.Frame` -- and by extension, a :class:`.History` --
contains many LoQS core objects or simple built-in Python objects,
there are several other kinds of objects that we commonly want
to store also.

This module serves as a place to store a lot of these objects.
"""

# Order here important, PauliFrame before all, QECCodePatch before PatchDict
from .pauliframe import PauliFrame
from .measurementoutcomes import MeasurementOutcomes
from .qeccodepatch import QECCodePatch
from .patchdict import PatchDict
