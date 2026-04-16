#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.BaseNoiseModel` and :class:`.BaseTimeDependentNoiseModel` definitions.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.reps import GateRep, InstrumentRep, RepTuple
from loqs.internal import Castable, Displayable


class BaseNoiseModel(Castable, Displayable):
    """Base class for an object that holds noisy operation specifications.

    This class is primarily designed to translate between a circuit description
    and operations that will be applied to a quantum state.
    """

    name: ClassVar[str]
    """Name of circuit backend"""

    CACHE_ON_SERIALIZE: ClassVar[bool] = True

    def __str__(self) -> str:
        return f"Physical {self.name} noise model\n"

    @property
    @abstractmethod
    def gate_keys(self) -> list:
        """Gate keys this model can take in circuits."""
        pass

    @property
    @abstractmethod
    def instrument_keys(self) -> list:
        """Instrument keys this model can take in circuits."""
        pass

    @property
    @abstractmethod
    def output_gate_reps(self) -> list[GateRep]:
        """Gate reps this model can output."""
        pass

    @property
    @abstractmethod
    def output_instrument_reps(self) -> list[InstrumentRep]:
        """Instrument reps this model can output."""
        pass

    @abstractmethod
    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gatereps: Sequence[GateRep],
        instreps: Sequence[InstrumentRep],
    ) -> list[RepTuple]:
        """Get list of operator representations that can be applied.

        Parameters
        ----------
        circuit:
            Physical circuit to get the representations for

        gatereps:
            Output representations for gate operations.
            For more details, look at (GateRep)[api:loqs.backends.reps.GateRep].

        instreps:
            Output representations for instrument operations.
            For more details, look at (InstrumentRep)[api:loqs.backends.reps.InstrumentRep].

        Returns
        -------
        list
            List of operation representations for the circuit

        REVIEW_SPHINX_REFERENCE
        """
        pass


class TimeDependentBaseNoiseModel(BaseNoiseModel):
    """Base class for an object that holds possibly *time-dependent* noisy operations.

    This class additionally holds the current time, as well as gate/instrument durations
    so that it can update the time as operations are applied.
    """

    current_time: float = 0.0
    """The current simulation time, according to the model."""

    _local_layer_duration: float = 0.0
    """The time taken for the current layer.

    This is mostly a helper variable that keeps track of the longest
    gate duration in a layer so that we update the time after each
    layer properly.
    """

    def add_gate_duration_to_layer(self, gate_duration):
        """Add a gate duration to the current layer duration.

        This method updates the current layer duration by taking the maximum
        of the existing layer duration and the provided gate duration.

        Parameters
        ----------
        gate_duration : float
            Duration of the gate to add to the current layer.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        self._local_layer_duration = max(
            self._local_layer_duration, gate_duration
        )

    def add_layer_duration_to_current_time(self):
        """Add the current layer duration to the total simulation time.

        This method updates the current simulation time by adding the duration
        of the current layer, then resets the layer duration to zero.

        Notes
        -----
        REVIEW_NO_DOCSTRING: This docstring was auto-generated for a function that
        previously had no documentation. Please review and update as needed.
        """
        self.current_time += self._local_layer_duration
        self._local_layer_duration = 0.0

    @abstractmethod
    def get_gate_duration(self, gate_label) -> int | float:
        """Get the gate duration from a gate label.

        Derived classes should implement this.
        """
        pass

    @abstractmethod
    def get_instrument_duration(self, inst_label) -> int | float:
        """Get the instrument duration from an instrument label.

        Derived classes should implement this.
        """
        pass
