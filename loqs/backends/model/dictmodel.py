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
from typing import ClassVar, Literal, TypeAlias, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit, ListPhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    RepTuple,
    ConcreteGateReps,
    ConcreteGateRep,
    ConcreteInstrumentReps,
)
from loqs.internal import SeqCastable
from loqs.internal.serializable import Serializable


T = TypeVar("T", bound="DictNoiseModel")

# Type aliases for static type checking
DictModelCastableTypes: TypeAlias = BaseNoiseModel | tuple[Mapping, Mapping]
"""Types of objects this backend can cast to dict models"""

MemberLabel: TypeAlias = str | tuple[str, tuple[str | int, ...]]


class DictNoiseModel(BaseNoiseModel, SeqCastable):
    """Model backend for handling generic operation dicts."""

    name: ClassVar[str] = "gate dict"
    gate_dict: dict[MemberLabel, RepTuple]
    inst_dict: dict[MemberLabel, RepTuple]

    SERIALIZE_ATTRS = ["gate_dict", "inst_dict", "_gatereps", "_instreps"]

    def __init__(  # noqa: C901
        self,
        model_or_dicts: DictModelCastableTypes,
        gatereps: Sequence[GateRep] = (GateRep.QSIM_SUPEROPERATOR,),
        instreps: Sequence[InstrumentRep] = (InstrumentRep.ZBASIS_PROJECTION,),
        gaterep_array_cast_rep: GateRep = GateRep.QSIM_SUPEROPERATOR,
        instrep_cast_reset: Literal[0, 1, None] = None,
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

        # NOTE: We set self.gate_dict and self.inst_dict at the end of this
        #  function. The next two variables are like gate_dict and inst_dict,
        #  but have more lax types.
        gate_dict: Mapping[MemberLabel, ConcreteGateRep | RepTuple] = {}
        #  ^ ConcreteGateRep is not so different from RepTuple, but it's different
        #    enough to matter.
        inst_dict: Mapping[MemberLabel, object] = {}
        # ^ The value type is just `object` because the set of all types that
        #   we can cast to an InstrumentRep in the current context (where we can
        #   rely on arguments `instrep` and `instrep_cast_include_outcomes`)
        #   is extremely broad.
        #

        if isinstance(model_or_dicts, DictNoiseModel):
            gate_dict = model_or_dicts.gate_dict.copy()
            inst_dict = model_or_dicts.inst_dict.copy()
        elif isinstance(model_or_dicts, PyGSTiNoiseModel):
            for gate_key in model_or_dicts.gate_keys:
                label = (gate_key.name, gate_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                gate_dict[label] = model_or_dicts.get_reps(
                    circ, gatereps=gatereps, instreps=instreps
                )[0][0]

            for inst_key in model_or_dicts.instrument_keys:
                label = (inst_key.name, inst_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                inst_dict[label] = model_or_dicts.get_reps(
                    circ, gatereps=gatereps, instreps=instreps
                )[0][0]

        elif isinstance(model_or_dicts, tuple) and len(model_or_dicts) == 2:
            gate_dict = dict(model_or_dicts[0])
            inst_dict = dict(model_or_dicts[1])
        else:
            raise TypeError(
                "Can only other NoiseModels or a 2-tuple of gate/inst dicts"
            )

        self._gatereps = list(gatereps)
        self._instreps = list(instreps)

        def convert_to_gatereptuple(gr, qubits) -> RepTuple:
            if not isinstance(gr, RepTuple):
                if isinstance(gr, np.ndarray):
                    # matrix for dense rep
                    return RepTuple(gr, qubits, gaterep_array_cast_rep)
                elif isinstance(gr, str):
                    return RepTuple(gr, qubits, GateRep.STIM_CIRCUIT_STR)
                elif isinstance(gr, (tuple, list)):
                    if ConcreteGateReps.sequence_is_krausop_rep(gr):
                        gr = tuple(tuple(el) for el in gr)  # cast to immutable
                        return RepTuple(gr, qubits, GateRep.KRAUS_OPERATORS)
                    elif ConcreteGateReps.sequence_is_probabilisticstim_rep(
                        gr
                    ):
                        gr = tuple(tuple(el) for el in gr)  # cast to immutable
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
        for k, gr in gate_dict.items():
            qubits = tuple() if isinstance(k, str) else k[1]
            gate_dict[k] = convert_to_gatereptuple(gr, qubits)

        # Run through instrument dict and upgrade everything to RepTuples
        for k, ir in inst_dict.items():
            qubits = tuple() if isinstance(k, str) else k[1]

            if not isinstance(ir, RepTuple) and isinstance(ir, str):
                rt = RepTuple(ir, qubits, InstrumentRep.STIM_CIRCUIT_STR)
                inst_dict[k] = rt

            elif ConcreteInstrumentReps.is_zbasis_projection_rep(ir):
                rt = RepTuple(ir, qubits, InstrumentRep.ZBASIS_PROJECTION)  # type: ignore
                inst_dict[k] = rt

            elif isinstance(ir, (tuple, list)) and len(ir) == 2:
                assert InstrumentRep.ZBASIS_PRE_POST_OPERATIONS in instreps, (
                    "Detected two ops for a pre/post operation instrument, but "
                    + "ZBASIS_PRE_POST_OPERATIONS not passed as a valid instrument rep"
                )
                concrep = (
                    instrep_cast_reset,
                    instrep_cast_include_outcomes,
                    convert_to_gatereptuple(ir[0], qubits),
                    convert_to_gatereptuple(ir[1], qubits),
                )
                inst_dict[k] = RepTuple(
                    concrep,
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
                new_ir = (
                    {
                        k: convert_to_gatereptuple(v, qubits)
                        for k, v in ir.items()
                    },
                    instrep_cast_include_outcomes,
                )
                inst_dict[k] = RepTuple(
                    new_ir,
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

        self.gate_dict: dict[MemberLabel, RepTuple] = gate_dict  # type: ignore
        self.inst_dict: dict[MemberLabel, RepTuple] = inst_dict  # type: ignore

        # TODO: Crosstalk specification?
        return

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
    def from_decoded_attrs(cls: type[T], attr_dict: Mapping) -> T:
        gate_dict = attr_dict["gate_dict"]
        inst_dict = attr_dict["inst_dict"]
        gatereps = [GateRep(v) for v in attr_dict["_gatereps"]]
        instreps = [InstrumentRep(v) for v in attr_dict["_instreps"]]
        return cls((gate_dict, inst_dict), gatereps, instreps)
