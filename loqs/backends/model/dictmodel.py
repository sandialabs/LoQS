""":class:`.DictNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit, ListPhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep, RepTuple


T = TypeVar("T", bound="DictNoiseModel")

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
        instrep: InstrumentRep = InstrumentRep.ZBASIS_PROJECTION,
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
        self.gate_dict: dict[tuple[str, tuple[str | int, ...]], object] = {}
        self.inst_dict: dict[tuple[str, tuple[str | int, ...]], object] = {}
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

    def __hash__(self) -> int:
        return hash(
            (
                self.hash(self.gate_dict),
                self.hash(self.inst_dict),
                self._gaterep.value,
                self._instrep.value,
            )
        )

    @property
    def gate_keys(self) -> list:
        """Gate keys this model can take in circuits."""
        return list(self.gate_dict.keys())

    @property
    def instrument_keys(self) -> list:
        """Instrument keys this model can take in circuits."""
        return list(self.inst_dict.keys())

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
    ) -> list[RepTuple]:
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
                reptype: GateRep | InstrumentRep = gaterep

                if rep is None:
                    # Failed, now look up in instruments
                    rep = self.inst_dict.get(label, None)
                    reptype = instrep

                assert rep is not None, f"Failed to look up {label}"

                reps.append(RepTuple(rep, label[1], reptype))
        return reps

    @classmethod
    def _from_serialization(
        cls: type[T], state: Mapping, serial_id_to_obj_cache=None
    ) -> T:
        # Not worth caching below this object (i.e. don't pass cache on)
        gate_dict = cls.deserialize(state["gate_dict"])
        assert isinstance(gate_dict, dict)
        inst_dict = cls.deserialize(state["inst_dict"])
        assert isinstance(inst_dict, dict)
        gaterep = GateRep(state["_gaterep"])
        instrep = InstrumentRep(state["_instrep"])
        return cls((gate_dict, inst_dict), gaterep, instrep)

    def _to_serialization(self, hash_to_serial_id_cache=None) -> dict:
        # Not worth caching below this object (i.e. don't pass cache on)
        state = super()._to_serialization()
        state.update(
            {
                "gate_dict": self.serialize(self.gate_dict),
                "inst_dict": self.serialize(self.inst_dict),
                "_gaterep": self._gaterep.value,
                "_instrep": self._instrep.value,
            }
        )
        return state
