""":class:`PyGSTiNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.circuit.listcircuit import ListPhysicalCircuit
from loqs.backends.model import BaseNoiseModel, GateRep, InstrumentRep
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel


# Type aliases for static type checking
GateDictModelCastableTypes: TypeAlias = (
    BaseNoiseModel | tuple[Mapping, Mapping]
)
"""Types of objects this backend can cast to dict models"""


class DictNoiseModel(BaseNoiseModel):
    """Model backend for handling generic operation dicts."""

    name: ClassVar[str] = "gate dict"

    def __init__(
        self,
        model_or_dicts: GateDictModelCastableTypes,
        gaterep: GateRep = GateRep.PTM,
        instrep: InstrumentRep = InstrumentRep.ZBASISPROJECTION,
    ) -> None:
        """Initialize a generic gate dict model.

        Parameters
        ----------
        model_or_dicts:
            A model to convert or pair of dictionaries to use

        gaterep:
            Gate representation this model will return

        instrep:
            Instrument representation this model will return
        """
        self.gate_dict = {}
        self.inst_dict = {}
        if isinstance(model_or_dicts, DictNoiseModel):
            self.gate_dict = model_or_dicts.gate_dict.copy()
            self.inst_dict = model_or_dicts.inst_dict.copy()
        elif isinstance(model_or_dicts, PyGSTiNoiseModel):
            for gate_key in model_or_dicts.gate_keys:
                label = (gate_key.name, gate_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                self.gate_dict[label] = model_or_dicts.get_reps(
                    circ, gaterep=gaterep, instrep=instrep
                )[0][0]

            for inst_key in model_or_dicts.instrument_keys:
                label = (inst_key.name, inst_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                self.inst_dict[label] = model_or_dicts.get_reps(
                    circ, gaterep=gaterep, instrep=instrep
                )[0][0]

        elif isinstance(model_or_dicts, tuple) and len(model_or_dicts) == 2:
            self.gate_dict = dict(model_or_dicts[0])
            self.inst_dict = dict(model_or_dicts[1])
        else:
            raise TypeError(
                "Can only other NoiseModels or a 2-tuple of gate/inst dicts"
            )

        self._gaterep = gaterep
        self._instrep = instrep

        # TODO: Crosstalk specification?

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
        circuit = ListPhysicalCircuit.cast(circuit)

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
