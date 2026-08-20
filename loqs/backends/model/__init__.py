#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Noise model backend classes.

For LoQS, a noise model is a mapping from a circuit label (i.e. gate name and target qubits) into some operator representation
that can be applied to a state. We need some way to enforce that a model's output can act on a state.
[](api:GateRep) is the abstract base class for representations that can be used for gates.
Similarly, [](api:InstrumentRep) is the abstract base class for representations that can be used for quantum instruments (often used to describe midcircuit measurements).

The model backend interface is enforced by the abstract [](api:BaseNoiseModel) class, which generally has the capabilities:

- Property getters for:
  - Allowed input circuit types
  - Output representation types
- Representation getter that converts a circuit into a list of gate/instrument representations

The packages currently available as noise models:

- Native `dict`-based model via [](api:DictNoiseModel), which natively handles both
  [](api:ListPhysicalCircuit) and [](api:STIMPhysicalCircuit) inputs.
  It does not require `loqs[stim]` itself, but its [](api:STIMPhysicalCircuit)-specific
  behavior does require `loqs[stim]`.
- [](api:pygsti.models.explicitmodel.ExplicitOpModel) and
[](api:pygsti.models.implicit.ImplicitOpModel) via [](api:PyGSTiNoiseModel) (requires `loqs[pygsti]`)

!!! warning

    For backends that depend on optional third-party packages,
    it is recommended to not import from the module/class file directly.
    Instead, try to import from [](api:loqs.backends), which dynamically checks
    if that backend is available.

"""

from .basemodel import BaseNoiseModel, TimeDependentBaseNoiseModel
from .dictmodel import DictNoiseModel, build_legacy_stim_dict_model

from loqs.internal.legacy import install_legacy_module, make_legacy_construction_shim

# STIMDictNoiseModel was removed in v1.2: DictNoiseModel now natively
# handles STIM circuits (case/alias-insensitive command lookup included),
# so a separate STIM-specific subclass is no longer needed. This shim
# keeps live code still calling STIMDictNoiseModel(...) working (with a
# deprecation warning) instead of hard-failing on an unresolvable import.
STIMDictNoiseModel = make_legacy_construction_shim(
    "STIMDictNoiseModel",
    build=build_legacy_stim_dict_model,
    message="STIMDictNoiseModel is deprecated; DictNoiseModel now natively "
    "handles STIM circuits. Constructing a DictNoiseModel on your behalf "
    "for now.",
)
install_legacy_module(
    "loqs.backends.model.stimdictmodel",
    {"STIMDictNoiseModel": STIMDictNoiseModel},
)
