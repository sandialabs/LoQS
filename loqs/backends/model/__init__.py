#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Noise model backend classes.

For LoQS, a noise model is a mapping from a circuit label (i.e. gate name and target qubits) into some operator representation
that can be applied to a state. We need some way to enforce that a model's output can act on a state.
The [](api:GateRep) enum defines what types of representations can be used for gates.
Similarly, the [](api:InstrumentRep) enum defines what types of representations can be used for quantum instruments (often used to describe midcircuit measurements).

The model backend interface is enforced by the abstract [](api:BaseNoiseModel) class, which generally has the capabilities:

- Property getters for:
  - Allowed input circuit types
  - Output representation types
- Representation getter that converts a circuit into a list of gate/instrument representations (as a tuple of rep, target qubits, and rep type)

The packages currently available as noise models:

- Native `dict`-based model via [](api:DictNoiseModel)
    - There is a second `dict`-based model ([](api:STIMDictNoiseModel)) that handles STIM circuit strings as values.
  Long-term this will be deprecated into [](api:DictNoiseModel).
  It does not require `loqs[stim]` itself, but requires [](api:STIMPhysicalCircuit) inputs (which do require `loqs[stim]`).
- [pygsti.models.explicitmodel.ExplicitOpModel](api:pygsti.models.explicitmodel.ExplicitOpModel) and
[pygsti.models.implicit.ImplicitOpModel](api:pygsti.models.implicit.ImplicitOpModel) via [](api:PyGSTiNoiseModel) (requires `loqs[pygsti]`)

!!! warning

    For backends that depend on optional third-party packages,
    it is recommended to not import from the module/class file directly.
    Instead, try to import from [](api:loqs.backends), which dynamically checks
    if that backend is available.

"""

from .basemodel import BaseNoiseModel, TimeDependentBaseNoiseModel
from .dictmodel import DictNoiseModel
