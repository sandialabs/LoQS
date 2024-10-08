""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model import BaseNoiseModel, GateRep, InstrumentRep


T = TypeVar("T", bound="DiscreteErrorInjectionNoiseModel")

# Type aliases for static type checking
GateDictModelCastableTypes: TypeAlias = BaseNoiseModel
"""Types of objects this backend can cast"""


class DiscreteErrorInjectionNoiseModel(BaseNoiseModel):
    """Model backend for a model with injected discrete errors."""

    name: ClassVar[str] = "discrete noise injection"

    def __init__(
        self,
        initial_model: GateDictModelCastableTypes,
        trigger_on_count: int,
        error_circuit: BasePhysicalCircuit | None = None,
        noise_reps: Sequence | None = None,
        inject_before: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize a noise model that injects discrete errors.

        Parameters
        ----------
        initial_model:
            An initial model to use when looking up operator representations

        trigger_on_count:
            Which call to `get_rep` should the error be injected on.

        error_circuit:
            A circuit to use to generate the noise to inject. Provide this
            *or* `noise_reps`.

        noise_reps:
            Noise reps to use during injection. Provide this *or* `error_circuit`.

        inject_before:
            A flag for whether the noise should be injected before (True, default)
            or after the operator representations.
        """
        self.underlying_model = initial_model
        assert (error_circuit is None and noise_reps is not None) or (
            error_circuit is not None and noise_reps is None
        ), "One of `error_circuit` and `noise_reps` must be provided"
        self.error_circuit = error_circuit
        self.noise_reps = list(noise_reps) if noise_reps is not None else None
        self.trigger_on_count = trigger_on_count
        self.get_rep_counter = 0
        self.inject_before = inject_before
        self.verbose = verbose

    @property
    def gate_keys(self) -> list:
        return self.underlying_model.gate_keys

    @property
    def instrument_keys(self) -> list:
        return self.underlying_model.instrument_keys

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
        # Get normal operation
        reps = self.underlying_model.get_reps(
            circuit, gaterep=gaterep, instrep=instrep
        )

        noise_reps = []
        if self.trigger_on_count == self.get_rep_counter:
            if self.error_circuit is not None:
                # Use the model to look up the noise reps
                noise_reps = self.underlying_model.get_reps(
                    self.error_circuit, gaterep=gaterep, instrep=instrep
                )
            else:
                assert self.noise_reps is not None
                noise_reps = self.noise_reps

            if self.verbose:
                timing = "before" if self.inject_before else "after"
                print(
                    f"Injecting discrete error {noise_reps} {timing} {circuit}"
                )

        if self.inject_before:
            reps = noise_reps + reps
        else:
            reps += noise_reps

        # Increment our counter
        self.get_rep_counter += 1

        return reps

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        model = cls.deserialize(
            state["underlying_model"], serial_id_to_obj_cache
        )
        assert isinstance(model, BaseNoiseModel)
        error_circuit = cls.deserialize(
            state["error_circuit"], serial_id_to_obj_cache
        )
        assert isinstance(error_circuit, BasePhysicalCircuit)
        noise_reps = state["noise_reps"]
        trigger_on_count = state["trigger_on_count"]
        get_rep_counter = state["get_rep_counter"]
        inject_before = state["inject_before"]
        verbose = state["verbose"]

        obj = cls(
            model,
            trigger_on_count,
            error_circuit,
            noise_reps,
            inject_before,
            verbose,
        )
        obj.get_rep_counter = get_rep_counter

        return obj

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        state = super()._to_serialization()
        state.update(
            {
                "underlying_model": self.serialize(
                    self.underlying_model, hash_to_serial_id_cache
                ),
                "error_circuit": self.serialize(
                    self.error_circuit, hash_to_serial_id_cache
                ),
                "noise_reps": self.noise_reps,
                "trigger_on_count": self.trigger_on_count,
                "get_rep_counter": self.get_rep_counter,
                "inject_before": self.inject_before,
                "verbose": self.verbose,
            }
        )
        return state
