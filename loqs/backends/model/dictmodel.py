""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, TypeAlias

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.circuit.builtincircuit import BuiltinPhysicalCircuit
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.backends.model import BaseNoiseModel, GateRep, InstrumentRep


# Type aliases for static type checking
GateDictModelCastableTypes: TypeAlias = (
    BaseNoiseModel | tuple[Mapping[object, object], Mapping[object, object]]
)
"""Types of objects this backend can cast to dict models"""


class DictNoiseModel(BaseNoiseModel):
    """Model backend for handling generic operation dicts."""

    name: ClassVar[str] = "gate dict"

    def __init__(
        self,
        model: GateDictModelCastableTypes,
        gaterep: GateRep = GateRep.PTM,
        instrep: InstrumentRep = InstrumentRep.ZBASISPROJECTION,
    ) -> None:
        """Initialize a generic gate dict model.

        Parameters
        ----------
        model:
            A pyGSTi model to use when looking up operations
        """
        self.gate_dict = {}
        self.inst_dict = {}
        if isinstance(model, BaseNoiseModel):
            for gate_key in model.gate_keys:
                circ = BuiltinPhysicalCircuit([gate_key])
                self.gate_dict[gate_key] = model.get_reps(
                    circ, gaterep=gaterep, instrep=instrep
                )

            for inst_key in model.instrument_keys:
                circ = BuiltinPhysicalCircuit([inst_key])
                self.inst_dict[inst_key] = model.get_reps(
                    circ, gaterep=gaterep, instrep=instrep
                )
        elif isinstance(model, tuple) and len(model) == 2:
            self.gate_dict = dict(model[0])
            self.inst_dict = dict(model[1])
        else:
            raise TypeError(
                "Can only other NoiseModels or a 2-tuple of gate/inst dicts"
            )

        self._gaterep = gaterep
        self._instrep = instrep

        # TODO: Crosstalk specification?

    @property
    def input_circuit_types(self) -> list[type[BasePhysicalCircuit]]:
        return [BuiltinPhysicalCircuit, PyGSTiPhysicalCircuit]

    @property
    def output_gate_reps(self) -> list[GateRep]:
        return [self._gaterep]

    @property
    def output_instrument_reps(self) -> list[InstrumentRep]:
        return [self._instrep]

    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gaterep: GateRep,
        instrep: InstrumentRep,
    ) -> list:
        assert (
            gaterep == self._gaterep
        ), f"Dict model only has {self._gaterep} gates"
        assert (
            instrep == self._instrep
        ), f"Dict model only has {self._instrep} instruments"

        # Get builtin circuit for easy processing
        circuit = BuiltinPhysicalCircuit.cast(circuit)

        # Iterate through circuit and pull out representations
        reps = []
        for layer in circuit.circuit:
            for label in layer:
                # Try to look up in gates
                rep = self.gate_dict.get(label, None)
                reptype = gaterep

                if rep is None:
                    # Failed, now look up in instruments
                    rep = self.inst_dict.get(label, None)
                    reptype = instrep

                assert rep is not None, f"Failed to look up {label}"

                reps.append((rep, label[1], reptype))
        return reps
