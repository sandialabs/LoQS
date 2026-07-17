#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections.abc import Mapping

from loqs.backends.model.dictmodel import DictNoiseModel


class STIMDictNoiseModel(DictNoiseModel):
    """Decode-only compatibility shim.

    `STIMDictNoiseModel` no longer exists as a usable class:
    [](api:DictNoiseModel) now handles [](api:STIMPhysicalCircuit) circuits
    directly (see `dictmodel.py`'s `get_reps` registration for
    `STIMPhysicalCircuit`), which is what this class used to exist
    separately to do. This class can no longer be constructed; it exists
    solely so that `.json`/`.h5` files serialized before this
    consolidation, which recorded `class: "STIMDictNoiseModel"`, continue
    to decode correctly -- to a plain [](api:DictNoiseModel) instance, not
    an instance of this class.

    This file (and this decode-only shim) should be deleted the next time
    `SERIALIZATION_VERSION` (`loqs.internal.serializable`) is genuinely
    bumped for an unrelated reason. At that point,
    `STIMDictNoiseModel`'s old `(module, class)` location can be migrated
    to an `IMPORT_LOCATION_CHANGES_BY_VERSION` entry redirecting it to
    [](api:DictNoiseModel) instead (the more natural mechanism for a
    genuine class relocation) at no extra marginal cost, since bumping
    `SERIALIZATION_VERSION` already requires auditing every
    `elif version == 1:` format-compatibility check in
    `loqs.internal.encoder.jsonencoder` regardless of this shim.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "STIMDictNoiseModel is deprecated; use DictNoiseModel directly "
            "(it now handles STIM circuits natively)."
        )

    @classmethod
    def _from_decoded_attrs(cls, attr_dict: Mapping) -> DictNoiseModel:
        return DictNoiseModel._from_decoded_attrs(attr_dict)
