""":class:`.DictNoiseModel` definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings
import numpy as np
from typing import ClassVar, TypeVar

from loqs.backends.circuit import BasePhysicalCircuit, STIMPhysicalCircuit
from loqs.backends.model.dictmodel import (
    DictNoiseModel,
    DictModelCastableTypes,
)
from loqs.backends.reps import GateRep, InstrumentRep, RepTuple


T = TypeVar("T", bound="DictNoiseModel")


class STIMDictNoiseModel(DictNoiseModel):
    """Model backend for handling generic operation dicts for STIM.

    This functionality should ideally by pulled into
    :class:`.DictNoiseModel`, but to make quick progress,
    we are making a derived class that can handle a
    :class:`.StimPhysicalCircuit` more naturally.
    """

    name: ClassVar[str] = "STIM gate dict"

    def __init__(
        self,
        model_or_dicts: DictModelCastableTypes,
        gatereps: Sequence[GateRep] = [GateRep.STIM_CIRCUIT_STR],
        instreps: Sequence[InstrumentRep] = [InstrumentRep.STIM_CIRCUIT_STR],
        gaterep_array_cast_rep: GateRep | None = None,
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
        if isinstance(model_or_dicts, STIMDictNoiseModel):
            self.gate_dict = model_or_dicts.gate_dict.copy()
            self.inst_dict = model_or_dicts.inst_dict.copy()
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
                if isinstance(gr, str):
                    return RepTuple(gr, qubits, GateRep.STIM_CIRCUIT_STR)
                elif isinstance(gr, (tuple, list)) and all(
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

        ## NOTE: This is a reduced support compared to DictNoiseModel
        assert all([gr in self._gatereps for gr in gatereps])
        assert all([ir in self._instreps for ir in instreps])
        if gaterep_array_cast_rep is not None:
            warnings.warn(
                "gaterep_array_cast_rep is set, but this option is not used by STIMDictNoiseModel"
            )

        # Run through gates and upgrade everything to RepTuples
        for k, gr in self.gate_dict.items():
            # By convention, choose that STIM commands should be uppercase
            if isinstance(k, str):
                name = k.upper()
                qubits = tuple()
                label = name
            else:
                name = k[0].upper()
                qubits = k[1]
                label = (name, qubits)
            self.gate_dict[label] = convert_to_gatereptuple(gr, qubits)

        # Run through instrument dict and upgrade everything to RepTuples
        for k, ir in self.inst_dict.items():
            # By convention, choose that STIM commands should be uppercase
            if isinstance(k, str):
                name = k.upper()
                qubits = tuple()
                label = name
            else:
                name = k[0].upper()
                qubits = k[1]
                label = (name, qubits)
            if not isinstance(ir, RepTuple) and isinstance(ir, str):
                self.inst_dict[label] = RepTuple(
                    ir, qubits, InstrumentRep.STIM_CIRCUIT_STR
                )
            elif (
                not isinstance(ir, RepTuple)
                and isinstance(ir, (tuple, list))
                and len(ir) == 2
                and (ir[0] is None or isinstance(ir[0], int))
                and isinstance(ir[1], bool)
            ):
                self.inst_dict[label] = RepTuple(
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
                self.inst_dict[label] = RepTuple(
                    (
                        instrep_cast_reset,
                        instrep_cast_include_outcomes,
                        convert_to_gatereptuple(ir[0], qubits),
                        convert_to_gatereptuple(ir[1], qubits),
                    ),
                    qubits,
                    InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
                )
            else:
                assert isinstance(
                    ir, RepTuple
                ), f"{ir} failed to upgrade to a RepTuple"
                assert (
                    ir.reptype in instreps
                ), f"Provided {ir} but reptype not in instreps"

    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gatereps: Sequence[GateRep],
        instreps: Sequence[InstrumentRep],
    ) -> list[RepTuple]:
        assert isinstance(
            circuit, STIMPhysicalCircuit
        ), "Only designed for STIM circuits"

        # Iterate through circuit and pull out representations
        reps = []
        for line in circuit._unroll_repeats().split("\n"):
            entries = line.split()

            if len(entries) == 0:
                # Empty line, but pass it on as a comment so full circuit has same formatting
                reps.append(RepTuple(line, tuple(), GateRep.STIM_CIRCUIT_STR))

            # Some commands can have parameters
            # Strip those so we can check base command name
            command = entries[0].split("(")[0]

            if command not in circuit._stim_gates:
                # We don't want to just ignore this, pass it on as a dummy reptuple
                # so that it will be included in the full circuit, but
                # we don't need to handle it here.
                # The empty qubit label will stand for this dummy/comment type rep
                reptuple = RepTuple(line, tuple(), GateRep.STIM_CIRCUIT_STR)
                reps.append(reptuple)

                # We'll warn if this is a noise channel, just FYI since we are
                # adding more noise
                if command in circuit._stim_noise_channels:
                    warnings.warn(
                        f"Noise channel '{line}' detected, but STIMDictNoiseModel is also adding noise"
                    )
            else:
                # This is a gate!
                # First check if measure had noise applied and warn
                if (
                    command in circuit._stim_measure_reset_gates
                    and "(" in entries[0]
                ):
                    warnings.warn(
                        f"Measure noise '{entries[0]}' detected, but STIMDictNoiseModel is also adding noise"
                    )

                mapped_qubits = []
                for qidx in entries[1:]:
                    negated = qidx.startswith("!")
                    qlabel = circuit.qubit_labels[int(qidx.strip("!"))]
                    mapped_qubits.append(f"{'!' if negated else ''}{qlabel}")

                # Put these in a tuple form commensurate with dict keys
                if command in circuit._stim_twoq_gates:
                    qubit_tuples = [
                        (mapped_qubits[i], mapped_qubits[i + 1])
                        for i in range(0, len(mapped_qubits), 2)
                    ]
                else:
                    # Otherwise everything is single qubit action
                    qubit_tuples = [(q,) for q in mapped_qubits]

                for qt in qubit_tuples:
                    label = (command.upper(), qt)

                    # Try to look up in gates
                    reptuple = self.gate_dict.get(label, None)

                    if reptuple is None:
                        # If that failed, check for generic name only
                        reptuple = self.gate_dict.get(command, None)
                        if reptuple is not None:
                            assert isinstance(reptuple, RepTuple)
                            reptuple = RepTuple(
                                reptuple.rep, qt, reptuple.reptype
                            )

                    if reptuple is None:
                        # Failed, now look up in instruments
                        reptuple = self.inst_dict.get(label, None)

                    if reptuple is None:
                        # If that failed, check for generic name only
                        reptuple = self.inst_dict.get(command, None)
                        if reptuple is not None:
                            assert isinstance(reptuple, RepTuple)
                            reptuple = RepTuple(
                                reptuple.rep, qt, reptuple.reptype
                            )

                    assert reptuple is not None, f"Failed to look up {label}"
                    assert isinstance(reptuple, RepTuple)

                    reps.append(reptuple)
        return reps
