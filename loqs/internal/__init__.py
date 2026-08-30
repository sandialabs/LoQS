#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.2                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Utility classes and functions for LoQS."""

import os
import socket

from .serializable import (
    Serializable,
    SERIALIZATION_VERSION,
    IncorrectDecodableTypeError,
    MisformedDecodableError,
)

# Must be after Serializable
from .displayable import Displayable


def worker_id() -> str:
    """Return this process's `hostname_pid` worker identity string, used to
    key per-writer checkpoint files across LoQS's parallel dispatch
    mechanisms."""
    return f"{socket.gethostname()}_{os.getpid()}"
