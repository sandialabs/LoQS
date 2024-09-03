""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, TypeAlias

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, GateRep, InstrumentRep


# Type aliases for static type checking
GateDictModelCastableTypes: TypeAlias = BaseNoiseModel
"""Types of objects this backend can cast"""


class DiscreteErrorInjectionNoiseModel(BaseNoiseModel):
    """Model backend for a model with injected discrete errors."""

    name: ClassVar[str] = "discrete noise injection"

    def __init__(
        self,
        initial_model: GateDictModelCastableTypes,
        error_circuit: (
            BasePhysicalCircuit
            | Callable[[BasePhysicalCircuit], BasePhysicalCircuit]
            | None
        ) = None,
        error_occurrences: Sequence[int] | None = None,
        occurrence_filters: (
            Sequence[BasePhysicalCircuit]
            | Callable[[BasePhysicalCircuit], bool]
            | None
        ) = None,
        occurrence_counter: int = 0,
        inject_after: bool = True,
    ) -> None:
        """Initialize a noise model that injects discrete errors.

        Parameters
        ----------
        initial_model:
            An initial model to use when looking up operator representations

        error_circuit:
            A circuit to use to generate the noise to inject. Can be a circuit,
            a function which takes the circuit passed to get_reps and returns a new circuit,
            or None (in which case no noise is injected).

        error_occurrences:
            Which occurrences (i.e. calls to get_reps) should have noise injected to them.
            The default is None (in which case no noise is injected).

        occurrence_filters:
            A filter for which calls to get_reps should count towards occurrences. Can be a circuit
            (which must match the input to get_reps exactly to count as an occurrence), a function
            that takes the input to get_reps and returns True/False, or None (the default, in which
            case all calls to get_reps count as occurrences).

        occurrence_counter:
            The counter for the number of calls to get_reps so far. Should probably be left
            to the default 0, but can be set to a different value for advanced usecases, i.e.
            this model is being initialized partway through a simulation and we want to note that
            some calls to get_reps have already happened.

        inject_after:
            A flag for whether the noise should be injected after (True, default) or before
            the operator representations.
        """
        self.underlying_model = initial_model
        self.error_occurrences = (
            [] if error_occurrences is None else error_occurrences
        )
        self.occurrence_counter = occurrence_counter
        self.inject_after = inject_after

        if error_circuit is None:
            self.error_circuit_fn = lambda x: None
        elif isinstance(error_circuit, BasePhysicalCircuit):
            self.error_circuit_fn = lambda x: error_circuit
        else:
            self.error_circuit_fn = error_circuit

        if occurrence_filters is None:
            self.filter_fn = lambda x: True
        elif isinstance(occurrence_filters, Callable):
            self.filter_fn = occurrence_filters
        else:
            self.filter_fn = lambda x: x in occurrence_filters

    @property
    def output_gate_reps(self) -> list[GateRep]:
        return self.underlying_model.output_gate_reps

    @property
    def output_instrument_reps(self) -> list[InstrumentRep]:
        return self.underlying_model.output_instrument_reps

    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gaterep: GateRep,
        instrep: InstrumentRep,
    ) -> list:
        # Increment our occurrence counter (if our circuit passes the filter)
        if self.filter_fn(circuit):
            self.occurrence_counter += 1

        # Get normal operation
        reps = self.underlying_model.get_reps(
            circuit, gaterep=gaterep, instrep=instrep
        )

        # Get error to inject
        error_circuit = self.error_circuit_fn(circuit)

        # If this is not one of our noisy occurrences or there was no error, shortcut exit
        if (
            self.occurrence_counter not in self.error_occurrences
            or error_circuit is None
        ):
            return reps

        # Get reps and inject before/after depending on flag
        noise_reps = self.underlying_model.get_reps(
            error_circuit, gaterep=gaterep, instrep=instrep
        )
        if self.inject_after:
            reps += noise_reps
        else:
            reps = noise_reps + reps

        return reps
