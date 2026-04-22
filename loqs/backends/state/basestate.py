#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################



from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
import textwrap
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends.model.basemodel import GateRep, InstrumentRep
from loqs.backends.reps import RepTuple
from loqs.internal import Castable, Displayable


# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="BaseQuantumState")

OutcomeDict: TypeAlias = dict[str | int, list[int]]
"""A type alias for outcome dictionaries."""


class BaseQuantumState(Castable, Displayable):
    """Base class for an object that holds a (physical) quantum state."""

    name: ClassVar[str]
    """Name of state backend"""

    _CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __str__(self) -> str:
        s = f"Physical {self.name} state:\n"
        s += textwrap.indent(str(self.state), "  ")
        return s

    # Provide a hook to deepcopy that avoids default pickling route
    # Often states will have a C/C++/Cython binding that doesn't work via pickling,
    # so we must leverage the object's non-default copy behavior
    def __deepcopy__(self: T, memo) -> T:
        return self.copy()

    @property
    @abstractmethod
    def state(self) -> object:
        """Getter of the underlying quantum state."""
        pass

    @property
    @abstractmethod
    def input_reps(self) -> list[GateRep | InstrumentRep]:
        """Gate and instrument reps this state can take as input.
        
        Returns
        -------
        list[GateRep | InstrumentRep]
            List of operation representation types that this quantum state backend
            can process and apply.
        """
        pass

    @abstractmethod
    def apply_reps_inplace(self, reps: Sequence[RepTuple]) -> OutcomeDict:
        """Apply the reps to the state in-place.

        Parameters
        ----------
        reps:
            Operator representations to apply

        Returns
        -------
        OutcomeDict
            Dictionary of outcomes. Can be empty if no
            measurements were performed.
        """
        pass

    @abstractmethod
    def apply_reps(self: T, reps: Sequence[RepTuple]) -> tuple[T, OutcomeDict]:
        """Apply the reps to the state in-place.

        Parameters
        ----------
        reps:
            See [apply_reps_inplace](api:BaseQuantumState.apply_reps_inplace).

        Returns
        -------
        BaseQuantumState, OutcomeDict
            A copy of the state with reps applied, and
            dictionary of outcomes. Outcomes can be empty if no
            measurements were performed.
        """
        new_state = self.copy()
        outputs = new_state.apply_reps_inplace(reps)
        return new_state, outputs

    @abstractmethod
    def copy(self: T) -> T:
        """Deep-copy a state object.

        Returns
        -------
            Copied state
        """
        pass

    @abstractmethod
    def reset_seed(self, new_seed: int | None) -> None:
        """Reset the RNG seed."""
        pass
