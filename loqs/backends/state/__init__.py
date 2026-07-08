#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Quantum state backends.

For LoQS, a quantum state represents the state of a group of physical qubits.

The state backend interface is enforced by the abstract `BaseQuantumState` class, which generally has the capabilities:

- Property getters for:
    - The underlying state object
    - Allowed input gate/instrument representations
- Copy (i.e. deepcopy)
- Representation application that takes the list of gate/instrument representation and propagates the state forward in time (in-place and copy)

The packages currently available as quantum states:

- Native `NumPy`-based statevector
- `quantumsim.sparsedm` density matrix via [](api:QSimQuantumState) (requires `loqs[quantumsim]`)
- `stim.TableauSimulator` stabilizers via [](api:STIMQuantumState) (requires `loqs[stim]`)

!!! warning

    For backends that depend on optional third-party packages,
    it is recommended to not import from the module/class file directly.
    Instead, try to import from [](api:loqs.backends), which dynamically checks
    if that backend is available.

"""

from .basestate import BaseQuantumState, OutcomeDict
from .npsvstate import NumpyStatevectorQuantumState
