#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.0                                                                             #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

""":class:`.DictNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from typing import ClassVar, TypeAlias, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit, ListPhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import GateRep, InstrumentRep, RepTuple
from loqs.internal import SeqCastable


T = TypeVar("T", bound="DictNoiseModel")

# Type aliases for static type checking
DictModelCastableTypes: TypeAlias = BaseNoiseModel | tuple[Mapping, Mapping]
"""Types of objects this backend can cast to dict models"""


class DictNoiseModel(BaseNoiseModel, SeqCastable):
    """Model backend for handling generic operation dicts."""

    name: ClassVar[str] = "gate dict"

    def __init__(  # noqa: C901
        self,
        model_or_dicts: DictModelCastableTypes,
        gatereps: Sequence[GateRep] = [GateRep.QSIM_SUPEROPERATOR],
        instreps: Sequence[InstrumentRep] = [InstrumentRep.ZBASIS_PROJECTION],
        gaterep_array_cast_rep: GateRep = GateRep.QSIM_SUPEROPERATOR,
        instrep_cast_reset: int | None = None,
        instrep_cast_include_outcomes: bool = True,
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

        instrep_cast_include_outcomes:
            If :attr:`.InstrumentRep.ZBASIS_PRE_POST_OPERATIONS` values
            are being cast up to :class:`.RepTuples`, this will be used as
            the first argument of the rep, indicating which state to reset
            to (``0`` or ``1``) or whether to not reset (``None``, default).

        instrep_cast_include_outcomes:
            If :attr:`.InstrumentRep.ZBASIS_PRE_POST_OPERATIONS` or
            :attr:`.InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT` values are
            being cast up to :class:`.RepTuples`, this will be used as
            the second argument of the rep, indicating whether outcomes
            should be kept (``True``, default) or not (``False``).
        """
        self.gate_dict: dict[
            str | tuple[str, tuple[str | int, ...]], object
        ] = {}
        self.inst_dict: dict[
            str | tuple[str, tuple[str | int, ...]], object
        ] = {}
        if isinstance(model_or_dicts, DictNoiseModel):
            self.gate_dict = model_or_dicts.gate_dict.copy()
            self.inst_dict = model_or_dicts.inst_dict.copy()
        elif isinstance(model_or_dicts, PyGSTiNoiseModel):
            for gate_key in model_or_dicts.gate_keys:
                label = (gate_key.name, gate_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                self.gate_dict[label] = model_or_dicts.get_reps(
                    circ, gatereps=gatereps, instreps=instreps
                )[0][0]

            for inst_key in model_or_dicts.instrument_keys:
                label = (inst_key.name, inst_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                self.inst_dict[label] = model_or_dicts.get_reps(
                    circ, gatereps=gatereps, instreps=instreps
                )[0][0]

        elif isinstance(model_or_dicts, tuple) and len(model_or_dicts) == 2:
            self.gate_dict = dict(model_or_dicts[0])
            self.inst_dict = dict(model_or_dicts[1])
        else:
            raise TypeError(
                "Can only other NoiseModels or a 2-tuple of gate/inst dicts"
            )

        self._gatereps = list(gatereps)
        self._instreps = list(instreps)

        def convert_to_gatereptuple(gr, qubits):
            if not isinstance(gr, RepTuple):
                if isinstance(gr, np.ndarray):
                    # matrix for dense rep
                    return RepTuple(gr, qubits, gaterep_array_cast_rep)
                elif isinstance(gr, str):
                    return RepTuple(gr, qubits, GateRep.STIM_CIRCUIT_STR)
                elif isinstance(gr, (tuple, list)):
                    if all(
                        [
                            isinstance(el, (tuple, list))
                            and len(el) == 2
                            and isinstance(el[0], np.ndarray)
                            and (
                                isinstance(el[1], (float, int))
                                or el[1] is None
                            )
                            for el in gr
                        ]
                    ):
                        return RepTuple(gr, qubits, GateRep.KRAUS_OPERATORS)
                    elif all(
                        [
                            isinstance(el, (tuple, list))
                            and len(el) == 2
                            and isinstance(el[0], str)
                            and isinstance(el[1], (float, int))
                            for el in gr
                        ]
                    ):
                        return RepTuple(
                            gr, qubits, GateRep.PROBABILISTIC_STIM_OPERATIONS
                        )

            assert isinstance(
                gr, RepTuple
            ), f"{gr} failed to upgrade to a RepTuple"
            assert (
                gr.reptype in gatereps
            ), f"Provided {gr} but not provided gatereps"

            return gr

        # Run through gates and upgrade everything to RepTuples
        for k, gr in self.gate_dict.items():
            qubits = tuple() if isinstance(k, str) else k[1]
            self.gate_dict[k] = convert_to_gatereptuple(gr, qubits)

        # Run through instrument dict and upgrade everything to RepTuples
        for k, ir in self.inst_dict.items():
            qubits = tuple() if isinstance(k, str) else k[1]
            if not isinstance(ir, RepTuple) and isinstance(ir, str):
                self.inst_dict[k] = RepTuple(
                    ir, qubits, InstrumentRep.STIM_CIRCUIT_STR
                )
            elif (
                not isinstance(ir, RepTuple)
                and isinstance(ir, (tuple, list))
                and len(ir) == 2
                and (ir[0] is None or isinstance(ir[0], int))
                and isinstance(ir[1], bool)
            ):
                self.inst_dict[k] = RepTuple(
                    ir, qubits, InstrumentRep.ZBASIS_PROJECTION
                )
            elif (
                not isinstance(ir, RepTuple)
                and isinstance(ir, (tuple, list))
                and len(ir) == 2
            ):
                assert InstrumentRep.ZBASIS_PRE_POST_OPERATIONS in instreps, (
                    "Detected two ops for a pre/post operation instrument, but "
                    + "ZBASIS_PRE_POST_OPERATIONS not passed as a valid instrument rep"
                )
                # Assume this is a 2-tuple of PTMS that we need to turn it into a RepTuple
                self.inst_dict[k] = RepTuple(
                    (
                        instrep_cast_reset,
                        instrep_cast_include_outcomes,
                        convert_to_gatereptuple(ir[0], qubits),
                        convert_to_gatereptuple(ir[1], qubits),
                    ),
                    qubits,
                    InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
                )
            elif not isinstance(ir, RepTuple) and isinstance(ir, Mapping):
                assert (
                    InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT in instreps
                ), (
                    "Detected dict for a outcome-operation instrument, but "
                    + "ZBASIS_OUTCOME_OPERATION_DICT not passed as a valid instrument rep"
                )
                # Assume this is a dict of PTMS that we need to turn it into a RepTuple
                self.inst_dict[k] = RepTuple(
                    (
                        {
                            k: convert_to_gatereptuple(v, qubits)
                            for k, v in ir.items()
                        },
                        instrep_cast_include_outcomes,
                    ),
                    qubits,
                    InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
                )
            else:
                assert isinstance(
                    ir, RepTuple
                ), f"{ir} failed to upgrade to a RepTuple"
                assert (
                    ir.reptype in instreps
                ), f"Provided {ir} but reptype not in instreps"

        # TODO: Crosstalk specification?

    def __hash__(self) -> int:
        return hash(
            (
                self.hash(self.gate_dict),
                self.hash(self.inst_dict),
                tuple([gr.value for gr in self._gatereps]),
                tuple([ir.value for ir in self._instreps]),
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
        return self._gatereps

    @property
    def output_instrument_reps(self) -> list[InstrumentRep]:
        return self._instreps

    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gatereps: Sequence[GateRep],
        instreps: Sequence[InstrumentRep],
    ) -> list[RepTuple]:
        # Get builtin circuit for easy processing
        circuit = ListPhysicalCircuit.cast(circuit)

        # Iterate through circuit and pull out representations
        reps = []
        for layer in circuit.circuit:
            for label in layer:
                # Try to look up in gates
                reptuple = self.gate_dict.get(label, None)

                if reptuple is None:
                    # Also try to look up just by name
                    reptuple = self.gate_dict.get(label[0], None)
                    if reptuple is not None:
                        assert isinstance(reptuple, RepTuple)
                        reptuple = RepTuple(
                            reptuple.rep, label[1], reptuple.reptype
                        )

                if reptuple is None:
                    # Failed, now look up in instruments
                    reptuple = self.inst_dict.get(label, None)

                if reptuple is None:
                    # Also try to look up just by name
                    reptuple = self.inst_dict.get(label[0], None)
                    if reptuple is not None:
                        assert isinstance(reptuple, RepTuple)
                        reptuple = RepTuple(
                            reptuple.rep, label[1], reptuple.reptype
                        )

                assert reptuple is not None, f"Failed to look up {label}"
                assert isinstance(reptuple, RepTuple)

                reps.append(reptuple)
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
        gatereps = [GateRep(v) for v in state["_gatereps"]]
        instreps = [InstrumentRep(v) for v in state["_instreps"]]
        return cls((gate_dict, inst_dict), gatereps, instreps)

    def _to_serialization(
        self, hash_to_serial_id_cache=None, ignore_no_serialize_flags=False
    ) -> dict:
        # Not worth caching below this object (i.e. don't pass cache on)
        state = super()._to_serialization()
        state.update(
            {
                "gate_dict": self.serialize(self.gate_dict),
                "inst_dict": self.serialize(self.inst_dict),
                "_gatereps": [gr.value for gr in self._gatereps],
                "_instreps": [ir.value for ir in self._instreps],
            }
        )
        return state
