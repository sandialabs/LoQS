#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.1.1                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################


from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings
from typing import ClassVar, TypeVar, Literal

from loqs.backends import BasePhysicalCircuit, STIMPhysicalCircuit
from loqs.backends.model.dictmodel import (
    DictNoiseModel,
    MemberLabel,
    DictModelCastableTypes,
)
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    RepTuple,
    ConcreteGateReps,
    ConcreteInstrumentReps,
)

T = TypeVar("T", bound="DictNoiseModel")


_STIM_CIRCUIT_STR_REPTYPES = (GateRep.STIM_CIRCUIT_STR, InstrumentRep.STIM_CIRCUIT_STR)
"""Reptypes whose `.rep` is STIM circuit-string text (and therefore has
per-qubit index placeholders to combine), as opposed to e.g.
`InstrumentRep.ZBASIS_PROJECTION`'s `(reset, include_outcomes)` tuple, which
carries no qubit-specific text at all.
"""

_SUPPORTED_GATEREPS = (GateRep.STIM_CIRCUIT_STR, GateRep.PROBABILISTIC_STIM_OPERATIONS)
"""`GateRep` members `STIMDictNoiseModel` (unlike the more general
[](api:DictNoiseModel)) can actually produce -- see `convert_to_gatereptuple`
in `__init__`.
"""

_SUPPORTED_INSTREPS = (
    InstrumentRep.STIM_CIRCUIT_STR,
    InstrumentRep.ZBASIS_PROJECTION,
    InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
)
"""`InstrumentRep` members `STIMDictNoiseModel` (unlike the more general
[](api:DictNoiseModel)) can actually produce -- see the instrument dict
loop in `__init__`. Notably, `InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT`
is not supported.
"""


def _merge_common_rep(
    command: str,
    qt: tuple,
    generic: RepTuple,
    common: dict[str, RepTuple],
) -> None:
    """Merge a generic (name-only) dict entry into `common[command]`.

    For `GateRep.STIM_CIRCUIT_STR`/`InstrumentRep.STIM_CIRCUIT_STR` reps,
    generic templates are combined across multiple qubits by *appending*
    further trailing qubit indices to every line for each additional qubit,
    so a template must already reference its own qubit(s) as indices
    ``0..len(qt)-1`` (e.g. `"X 0"` for a single-qubit template, `"CNOT 0 1"`
    for a two-qubit one) before it is used for the first time here. Without
    this, the first qubit's index would silently never be added -- this
    raises a clear error instead of producing a `RepTuple` that claims to
    act on a qubit its own circuit string never actually references.

    For other reptypes (e.g. `InstrumentRep.ZBASIS_PROJECTION`), `.rep` has
    no per-qubit text to combine -- the same payload applies uniformly
    regardless of how many qubits are merged in (state backends apply it in
    a `for qbit in qubits: ...` loop), so merging is just concatenating the
    qubit tuples.
    """
    prev = common.get(command)

    if generic.reptype not in _STIM_CIRCUIT_STR_REPTYPES:
        prev_qubits: tuple = prev.qubits if prev is not None else tuple()
        common[command] = RepTuple(
            generic.rep, prev_qubits + qt, generic.reptype
        )
        return

    if prev is None:
        # First occurrence: the raw template must already reference its own
        # qubit(s) as indices 0..len(qt)-1, and is used exactly as registered
        # -- no further indices are appended for the qubit(s) it already
        # covers. Only later occurrences (below) append further trailing
        # indices for *additional* qubits.
        expected = [str(i) for i in range(len(qt))]
        for line in generic.rep.split("\n"):
            trailing = line.split()[-len(qt) :] if len(qt) else []
            if trailing != expected:
                raise ValueError(
                    f"Generic STIM circuit-string template for command "
                    f"{command!r} (line {line!r}) must already reference its "
                    f"own qubit(s) as index/indices {expected} before it can "
                    "be combined across multiple qubits, since additional "
                    "qubits are handled by appending further trailing "
                    f"indices; got trailing tokens {trailing!r}."
                )
        common[command] = RepTuple(generic.rep, qt, generic.reptype)
        return

    new_lines = [
        line + "".join(f" {len(prev.qubits) + i}" for i in range(len(qt)))
        for line in prev.rep.split("\n")
    ]
    common[command] = RepTuple(
        "\n".join(new_lines), prev.qubits + qt, generic.reptype
    )


def add_command_aliases(d: dict[MemberLabel, Any]) -> None:
    aliases = STIMPhysicalCircuit.stim_command_aliases

    need_aliasing = []
    for k in d:
        if isinstance(k, str) and k in aliases:
            need_aliasing.append(k)
        if isinstance(k, tuple) and k[0] in aliases:
            need_aliasing.append(k)

    for k in need_aliasing:
        if isinstance(k, str):
            aliased_k = aliases[k]
        elif isinstance(k, tuple):
            aliased_k = (aliases[k[0]],) + k[1:]
        d[aliased_k] = d[k]

    return


class STIMDictNoiseModel(DictNoiseModel):
    """Model backend for handling generic operation dicts for STIM.

    This functionality should ideally by pulled into
    [](api:DictNoiseModel), but to make quick progress,
    we are making a derived class that can handle a
    [](api:STIMPhysicalCircuit) more naturally.
    """

    name: ClassVar[str] = "STIM gate dict"

    def __init__(
        self,
        model_or_dicts: DictModelCastableTypes,
        gatereps: Sequence[GateRep] = (GateRep.STIM_CIRCUIT_STR,),
        instreps: Sequence[InstrumentRep] = (InstrumentRep.STIM_CIRCUIT_STR,),
        gaterep_array_cast_rep: GateRep | None = None,
        instrep_cast_reset: Literal[0, 1, None] = None,
        instrep_cast_include_outcomes: bool = True,
    ) -> None:
        # NOTE: We set self.gate_dict and self.inst_dict at the end of this
        # function. The next two variables are like gate_dict and inst_dict,
        # but have more lax types.
        gate_dict: Mapping[
            MemberLabel,
            RepTuple
            | ConcreteGateReps.STIM_CIRCUIT_STR_t
            | ConcreteGateReps.PROBABILISTIC_STIM_OPERATIONS_t,
        ] = {}
        inst_dict: Mapping[MemberLabel, object] = {}

        if isinstance(model_or_dicts, STIMDictNoiseModel):
            gate_dict = model_or_dicts.gate_dict.copy()
            inst_dict = model_or_dicts.inst_dict.copy()
        elif isinstance(model_or_dicts, tuple) and len(model_or_dicts) == 2:
            gate_dict = dict(model_or_dicts[0])
            inst_dict = dict(model_or_dicts[1])
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
                elif ConcreteGateReps.sequence_is_probabilisticstim_rep(gr):
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

        ## NOTE: This is a reduced support compared to DictNoiseModel
        assert all(gr in _SUPPORTED_GATEREPS for gr in gatereps), (
            f"STIMDictNoiseModel only supports {_SUPPORTED_GATEREPS} as "
            f"gatereps, got {gatereps}"
        )
        assert all(ir in _SUPPORTED_INSTREPS for ir in instreps), (
            f"STIMDictNoiseModel only supports {_SUPPORTED_INSTREPS} as "
            f"instreps, got {instreps}"
        )
        if gaterep_array_cast_rep is not None:
            warnings.warn(
                "gaterep_array_cast_rep is set, but this option is not used by STIMDictNoiseModel"
            )

        def promoted_key_and_qubits(k):
            # By convention, choose that STIM commands should be uppercase
            if isinstance(k, str):
                return k.upper(), tuple()
            else:
                return (k[0].upper(), k[1]), k[1]

        # Run through gates and upgrade everything to RepTuples
        gate_dict_unfiltered = gate_dict.copy()
        gate_dict.clear()
        for k, gr in gate_dict_unfiltered.items():
            k, qubits = promoted_key_and_qubits(k)
            gate_dict[k] = convert_to_gatereptuple(gr, qubits)

        # Run through instrument dict and upgrade everything to RepTuples
        inst_dict_unfiltered = inst_dict.copy()
        inst_dict.clear()
        for k, ir in inst_dict_unfiltered.items():
            k, qubits = promoted_key_and_qubits(k)

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

            else:
                assert isinstance(
                    ir, RepTuple
                ), f"{ir} failed to upgrade to a RepTuple"
                assert (
                    ir.reptype in instreps
                ), f"Provided {ir} but reptype not in instreps"
                inst_dict[k] = ir

        self.gate_dict: dict[MemberLabel, RepTuple] = gate_dict  # type: ignore
        self.inst_dict: dict[MemberLabel, RepTuple] = inst_dict  # type: ignore
        add_command_aliases(self.gate_dict)
        return

    def get_reps(
        self,
        circuit: STIMPhysicalCircuit,
        gatereps: Sequence[GateRep],
        instreps: Sequence[InstrumentRep],
    ) -> list[RepTuple]:
        """Get operation representations for a STIM circuit.

        This method processes a STIM circuit and generates a list of operation
        representations (RepTuples) that describe the circuit's operations,
        including gates and instruments, with appropriate noise models applied.

        This method handles STIM-specific circuit processing including:
        - Unrolling circuit repeats
        - Mapping qubit labels
        - Applying noise models from gate and instrument dictionaries
        - Combining common operations for efficient STIM processing
        - Warning about conflicting noise applications

        Parameters
        ----------
        circuit:
            The STIM circuit to process.

        gatereps:
            Sequence of gate representations that the output should use.

        instreps:
            Sequence of instrument representations that the output should use.

        Returns
        -------
        list[RepTuple]
            List of operation representations describing the circuit's operations
            with noise models applied.

        Raises
        ------
        AssertionError
            If the circuit is not a STIMPhysicalCircuit instance.
        """
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
                continue

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

                # When we are looking things up generically by name only,
                # we can combine all qubit tuples into a common command
                # This is mostly to make STIM print these circuits much nicer
                common = {}
                for qt in qubit_tuples:
                    is_common = False
                    label = (command.upper(), qt)

                    # Try to look up in gates
                    reptuple = self.gate_dict.get(label, None)

                    if reptuple is None:
                        # If that failed, check for generic name only
                        generic = self.gate_dict.get(command, None)
                        if generic is not None:
                            assert isinstance(generic, RepTuple)
                            is_common = True
                            _merge_common_rep(command, qt, generic, common)
                            reptuple = common[command]

                    if reptuple is None:
                        # Failed, now look up in instruments
                        reptuple = self.inst_dict.get(label, None)

                    if reptuple is None:
                        # If that failed, check for generic name only
                        generic = self.inst_dict.get(command, None)
                        if generic is not None:
                            assert isinstance(generic, RepTuple)
                            is_common = True
                            _merge_common_rep(command, qt, generic, common)
                            reptuple = common[command]

                    assert reptuple is not None, f"Failed to look up {label}"
                    assert isinstance(reptuple, RepTuple)

                    if not is_common:
                        reps.append(reptuple)

                # Add common commands to the rep
                reps.extend(common.values())

        return reps
