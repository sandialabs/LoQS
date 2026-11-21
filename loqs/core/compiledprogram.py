#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""[](api:CompiledProgram) definition."""

from __future__ import annotations

from dataclasses import dataclass

from loqs.internal import Displayable


@dataclass
class CompiledProgram(Displayable):
    """What an engine runs for a program: plain, serializable data.

    It is filled here by tracing one shot of the program, and later could
    be filled ahead of time by a static compiler. It holds no `History`
    index, closure or live object, and is keyed by the measurement
    record, which is the concatenation of the steps' outcome bits.

    Each entry of [](api:steps) is a `dict` with five keys:

    - `name` and `type`: those of the instruction that ran.
    - `outcome_labels`: the qubit labels behind the record bits the step
      wrote, in record order; an entry is `None` for a bit with no qubit
      label (an `MPP` bit).
    - `emitted`: a `dict` from a target name to what the step emitted for
      that target, so one step can carry several targets. The `"stim"`
      target holds the STIM text the step ran, and a step that emits
      nothing has an empty `dict`.
    - `reads`: a sorted list of the indices of earlier steps whose output
      the step collects, which records feed-forward as data flow. It is an
      over-approximation: every earlier step that set a key the step
      collects counts, and an in-place change to shared data isn't seen.

    A step may gain keys later, so read a step by key and never assume
    these five are all there is.
    """

    _SERIALIZE_ATTRS = ["qubit_labels", "steps"]

    qubit_labels: list
    """The program's qubit register.

    A target's internal index `i` in emitted text means `qubit_labels[i]`.
    """

    steps: list
    """One `dict` per executed instruction, in run order."""
