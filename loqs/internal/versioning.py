#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Per-version decode dispatch for [](api:Serializable)'s encoders."""

from __future__ import annotations

from typing import Any, Callable

_ALL_VERSIONED_DECODERS: list["VersionedDecoder"] = []
"""Every VersionedDecoder ever constructed, for completeness testing."""


class VersionedDecoder:
    """Registry mapping a serialization version to the decoder for that version's on-disk shape.

    Each registered decoder fully decodes its own version's shape straight
    to the final, current-codebase Python value -- there is no chaining
    between versions here (contrast `IMPORT_LOCATION_CHANGES_BY_VERSION`,
    which does chain, since a class's import location can move again in a
    later version; a given version's on-disk shape never does). Encode
    never needs this: it always writes the current `SERIALIZATION_VERSION`'s
    shape, never an old one.
    """

    def __init__(self, name: str):
        self._name = name
        self._decoders: dict[int, Callable] = {}
        _ALL_VERSIONED_DECODERS.append(self)

    def register(self, version: int):
        """Decorator registering `fn` as the decoder for `version`."""

        def decorator(fn):
            assert (
                version not in self._decoders
            ), f"{self._name} already has a decoder for version {version}"
            self._decoders[version] = fn
            return fn

        return decorator

    def alias(self, version: int, same_as: int) -> None:
        """Register `version` to reuse the decoder already registered for `same_as`."""
        assert (
            same_as in self._decoders
        ), f"{self._name} has no decoder for version {same_as} to alias"
        self.register(version)(self._decoders[same_as])

    def __call__(self, version: int, *args: Any, **kwargs: Any) -> Any:
        from loqs.internal.serializable import DecodableVersionError

        try:
            decoder = self._decoders[version]
        except KeyError:
            raise DecodableVersionError(
                f"No {self._name} decoder for version {version}"
            )
        return decoder(*args, **kwargs)
