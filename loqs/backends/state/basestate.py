#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3.1                                                                         #
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

import numpy as np
from numpy.random import SeedSequence

from loqs.backends.reps import GateRep, InstrumentRep, OperationRep
from loqs.internal import Displayable

# Generic type variable to stand-in for derived class below
T = TypeVar("T", bound="BaseQuantumState")

OutcomeDict: TypeAlias = dict[str | int, list[int]]
"""A type alias for outcome dictionaries."""


def as_seed_sequence(seed: int | SeedSequence | None) -> SeedSequence:
    """Turn a seed into a `SeedSequence`.

    A `SeedSequence` is returned as the same object. Anything else goes to
    `SeedSequence(seed)`, so None draws fresh entropy and a bad value raises
    NumPy's own error.
    """
    if isinstance(seed, SeedSequence):
        return seed
    return SeedSequence(seed)


def child_seed_sequence(parent: SeedSequence, *key: int) -> SeedSequence:
    """Derive the child of `parent` at `key`, independent of call order.

    The result equals what `parent.spawn()` would give for the matching
    index, but never touches the parent's spawn counter.
    """
    return SeedSequence(
        parent.entropy,
        spawn_key=tuple(parent.spawn_key) + tuple(key),
        pool_size=parent.pool_size,
    )


class BaseQuantumState(Displayable):
    """Base class for an object that holds a (physical) quantum state."""

    name: ClassVar[str]
    """Name of state backend"""

    seed_sequence: SeedSequence
    """Seed sequence the state's generators are built from."""

    _rng: np.random.Generator
    """Random number generator."""

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
    def input_reps(self) -> list[type[GateRep | InstrumentRep]]:
        """Gate and instrument rep classes this state can take as input.

        Returns
        -------
        list[type[GateRep | InstrumentRep]]
            List of operation representation classes that this quantum
            state backend can process and apply. Compatibility with a
            model's declared output representations is checked via
            `issubclass` (see [](api:is_rep_compatible)), so a state may
            declare a coarse-grained entry like `InstrumentRep` to accept
            any instrument representation.
        """
        pass

    @abstractmethod
    def apply_reps_inplace(self, reps: Sequence[OperationRep]) -> OutcomeDict:
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
    def apply_reps(
        self: T, reps: Sequence[OperationRep]
    ) -> tuple[T, OutcomeDict]:
        """Apply the reps to the state in-place.

        Parameters
        ----------
        reps:
            See [](api:BaseQuantumState.apply_reps_inplace).

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

    def reset_seed(self, new_seed: int | SeedSequence | None) -> None:
        """Reset the RNG seed.

        Parameters
        ----------
        new_seed:
            An integer, a NumPy `SeedSequence` (used as is), or None for
            fresh entropy. The state keeps the resulting sequence in
            `seed_sequence`, so a run can be replayed from it. A backend with
            generators besides `_rng` overrides this to seed them too.
        """
        self.seed_sequence = as_seed_sequence(new_seed)
        self._rng = np.random.default_rng(self.seed_sequence)
