#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""TODO"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from loqs.internal import Displayable

T = TypeVar("T", bound="SyndromeLabel")


SyndromeLabelCastableTypes: TypeAlias = (
    "str | tuple[str] | tuple[str, int] | tuple[str, int, int] | SyndromeLabel"
)
"""Objects that can be cast to [](api:SyndromeLabel) objects."""


@dataclass
class SyndromeLabel(Displayable):
    """Label that indicates which past outcome was a syndrome bit."""

    _SERIALIZE_ATTRS = ["qubit_label", "frame_idx", "outcome_idx"]

    qubit_label: str | int
    """The qubit label."""

    frame_idx: int = -1
    """The frame index.

    Defaults to -1, i.e. the previous frame.
    """

    outcome_idx: int = 0
    """The outcome index.

    Defaults to 0, the first outcome on [](api:qubit_label).
    Could be >0 if multiple checks were measured on [](api:qubit_label).
    """

    @classmethod
    def from_raw(cls, obj: object) -> SyndromeLabel:
        """Build a [](api:SyndromeLabel) from a loosely-typed raw value.

        Several call sites hand this class a genuinely ambiguous raw
        blob -- a bare `str`, a variable-length tuple to unpack, or an
        already-built [](api:SyndromeLabel) -- which a constructor's
        positional-argument signature alone cannot disambiguate.

        Parameters
        ----------
        obj:
            A raw value that is either:
            - Already a [](api:SyndromeLabel) object
            - A kwarg dict that is passed into the constructor
            - A sequence of the arguments of the
            [](api:SyndromeLabel) constructor
            - A bare `qubit_label`

        Returns
        -------
            A [](api:SyndromeLabel) object
        """
        if isinstance(obj, SyndromeLabel):
            # We are already the correct class, perform no copy
            return obj
        elif isinstance(obj, Mapping):
            # Assume this is a kwarg dict, pass in all kwargs
            return cls(**obj)
        elif isinstance(obj, Sequence) and not isinstance(obj, str):
            # Assume this is a tuple of arguments, pass all in
            return cls(*obj)

        # Otherwise, assume this is the bare qubit_label
        return cls(obj)  # type: ignore
