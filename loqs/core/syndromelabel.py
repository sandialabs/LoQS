#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""TODO
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from loqs.internal import Castable, Displayable

T = TypeVar("T", bound="SyndromeLabel")


SyndromeLabelCastableTypes: TypeAlias = (
    "str | tuple[str] | tuple[str, int] | tuple[str, int, int] | SyndromeLabel"
)
"""Objects that can be cast to [](api:SyndromeLabel) objects."""


@dataclass
class SyndromeLabel(Castable, Displayable):
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

    Defaults to 0, the first outcome on :attr:`.qubit_label`.
    Could be >0 if multiple checks were measured on :attr:`.qubit_label`.
    """
