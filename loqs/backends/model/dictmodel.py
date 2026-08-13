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
from functools import singledispatchmethod
from types import NoneType
import copy
import warnings
import numpy as np
from typing import ClassVar, Literal, TypeAlias, TypeVar, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loqs.backends.model.pygstimodel import PyGSTiNoiseModel

from loqs.backends.circuit import BasePhysicalCircuit, ListPhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    KrausGateRep,
    OperationRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    StimCircuitPayloadMixin,
    ZBasisOutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
    convert as convert_rep,
)
from loqs.backends.reps.legacy import (
    upgrade_legacy_gaterep_tag,
    upgrade_legacy_instrumentrep_tag,
)
from loqs.internal import SeqCastable
from loqs.internal.serializable import Serializable

# `STIMPhysicalCircuit` is used opportunistically, not required: `dictmodel`
# doesn't need `stim` to be installed for the `ListPhysicalCircuit` case,
# it just knows how to handle a `STIMPhysicalCircuit` if one shows up (the
# same soft-dependency idiom already used in `stimcircuit.py`/`stimstate.py`
# -- see also `loqs.backends`'s module-level comment about import ordering,
# which this relies on to avoid a circular-import false negative).
if TYPE_CHECKING:
    from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
else:
    try:
        from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
    except ImportError:
        STIMPhysicalCircuit = None  # type: ignore

T = TypeVar("T", bound="DictNoiseModel")

# Type aliases for static type checking
DictModelCastableTypes: TypeAlias = BaseNoiseModel | tuple[Mapping, Mapping]
"""Types of objects this backend can cast to dict models"""

MemberLabel: TypeAlias = str | tuple[str, tuple[str | int, ...]]

_NON_ARRAY_GATEREPS: tuple[type[GateRep], ...] = (
    StimCircuitGateRep,
    KrausGateRep,
    ProbabilisticStimGateRep,
)
"""Candidate [](api:GateRep) classes tried (in order) for a raw gate value
that isn't a bare array. A bare array is handled separately (see
`DictNoiseModel.__init__`'s `gaterep_array_cast_rep`), since
[](api:UnitaryGateRep)/[](api:PTMGateRep)/[](api:QSimSuperopGateRep)
cannot be structurally distinguished from each other by shape alone.
"""


class DictNoiseModel(BaseNoiseModel, SeqCastable):
    """Model backend for handling generic operation dicts.

    This natively handles both [](api:ListPhysicalCircuit) circuits (via a
    simple per-label dictionary lookup) and [](api:STIMPhysicalCircuit)
    circuits (via a STIM-text-parsing traversal, registered as a
    `functools.singledispatchmethod` implementation of `get_reps` -- see
    that method) when `loqs[stim]` is installed.
    """

    name: ClassVar[str] = "gate dict"

    gate_dict: dict[MemberLabel, GateRep]
    """Mapping from labels to [](api:GateRep) for gate operations."""

    inst_dict: dict[MemberLabel, InstrumentRep]
    """Mapping from labels to [](api:InstrumentRep) for instrument operations."""

    _SERIALIZE_ATTRS = ["gate_dict", "inst_dict", "_gatereps", "_instreps"]

    def __init__(  # noqa: C901
        self,
        model_or_dicts: DictModelCastableTypes,
        gatereps: Sequence[type[GateRep]] = (QSimSuperopGateRep,),
        instreps: Sequence[type[InstrumentRep]] = (ZBasisProjectionInstrumentRep,),
        gaterep_array_cast_rep: type[GateRep] = QSimSuperopGateRep,
        instrep_cast_reset: Literal[0, 1, None] = None,
        instrep_cast_include_outcomes: bool = True,
    ) -> None:
        """Initialize a generic gate dict model.

        Parameters
        ----------
        model_or_dicts:
            A model to convert or pair of dictionaries to use

        gatereps:
            [](api:GateRep) classes this model will return

        instreps:
            [](api:InstrumentRep) classes this model will return

        gaterep_array_cast_rep:
            The [](api:GateRep) class used when casting bare matrices.
            A bare array cannot be unambiguously classified as e.g. a
            unitary vs. a Pauli-transfer matrix by shape alone, so this
            disambiguates explicitly rather than relying on structural
            matching.

        instrep_cast_reset:
            If [](api:ZBasisPrePostInstrumentRep) values are being cast up
            from raw values, this will be used as the `reset` field,
            indicating which state to reset to (`0` or `1`) or whether to
            not reset (`None`, default).

        instrep_cast_include_outcomes:
            If [](api:ZBasisPrePostInstrumentRep) or
            [](api:ZBasisOutcomeOperationDictInstrumentRep) values are
            being cast up from raw values, this will be used as the
            `include_outcome` field, indicating whether outcomes should be
            kept (`True`, default) or not (`False`).
        """

        # NOTE: We set self.gate_dict and self.inst_dict at the end of this
        #  function. The next two variables are like gate_dict and inst_dict,
        #  but have more lax types.
        gate_dict: dict[MemberLabel, Any] = {}
        inst_dict: dict[MemberLabel, Any] = {}

        if isinstance(model_or_dicts, DictNoiseModel):
            gate_dict = model_or_dicts.gate_dict.copy()
            inst_dict = model_or_dicts.inst_dict.copy()
        elif type(model_or_dicts).__name__ == "PyGSTiNoiseModel":
            pygsti_model: Any = model_or_dicts
            for gate_key in pygsti_model.gate_keys:
                label = (gate_key.name, gate_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                gate_dict[label] = pygsti_model.get_reps(
                    circ, gatereps=gatereps, instreps=instreps
                )[0][0]

            for inst_key in pygsti_model.instrument_keys:
                label = (inst_key.name, inst_key.qubits)
                circ = ListPhysicalCircuit([[label]])
                inst_dict[label] = pygsti_model.get_reps(
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

        def convert_to_gaterep(gr, qubits) -> GateRep:
            if isinstance(gr, GateRep):
                assert any(
                    isinstance(gr, cls) for cls in gatereps
                ), f"Provided {gr} but not provided gatereps"
                return gr
            if isinstance(gr, np.ndarray):
                # matrix for dense rep; None (unknown qubits) means "not attached yet".
                return gaterep_array_cast_rep(gr, () if qubits is None else qubits)
            return convert_rep(gr, _NON_ARRAY_GATEREPS, qubits)

        def convert_to_instrumentrep(ir, qubits) -> InstrumentRep:
            if isinstance(ir, InstrumentRep):
                assert any(
                    isinstance(ir, cls) for cls in instreps
                ), f"Provided {ir} but reptype not in instreps"
                return ir
            # A length-2 tuple/list is ambiguous between a (reset,
            # include_outcome) pair and a (pre_op, post_op) pair.
            looks_like_zbasis_projection = (
                isinstance(ir, (tuple, list))
                and len(ir) == 2
                and isinstance(ir[0], (int, NoneType))
                and isinstance(ir[1], bool)
            )
            if looks_like_zbasis_projection:
                return ZBasisProjectionInstrumentRep(ir[0], ir[1], qubits)
            if isinstance(ir, (tuple, list)) and len(ir) == 2:
                assert ZBasisPrePostInstrumentRep in instreps, (
                    "Detected two ops for a pre/post operation instrument, but "
                    + "ZBasisPrePostInstrumentRep not passed as a valid instrument rep"
                )
                pre_op = convert_to_gaterep(ir[0], qubits)
                post_op = convert_to_gaterep(ir[1], qubits)
                return ZBasisPrePostInstrumentRep(
                    instrep_cast_reset,
                    instrep_cast_include_outcomes,
                    pre_op,
                    post_op,
                    qubits,
                )
            if isinstance(ir, Mapping):
                assert ZBasisOutcomeOperationDictInstrumentRep in instreps, (
                    "Detected dict for a outcome-operation instrument, but "
                    + "ZBasisOutcomeOperationDictInstrumentRep not passed as a valid instrument rep"
                )
                outcome_ops = {
                    k: convert_to_gaterep(v, qubits) for k, v in ir.items()
                }
                return ZBasisOutcomeOperationDictInstrumentRep(
                    outcome_ops, instrep_cast_include_outcomes, qubits
                )
            return convert_rep(ir, StimCircuitInstrumentRep, qubits)

        # Run through gates and upgrade everything to GateReps
        for k, gr in gate_dict.items():
            qubits = None if isinstance(k, str) else k[1]
            gate_dict[k] = convert_to_gaterep(gr, qubits)

        # Run through instrument dict and upgrade everything to InstrumentReps
        for k, ir in inst_dict.items():
            qubits = None if isinstance(k, str) else k[1]
            inst_dict[k] = convert_to_instrumentrep(ir, qubits)

        self.gate_dict: dict[MemberLabel, GateRep] = gate_dict
        self.inst_dict: dict[MemberLabel, InstrumentRep] = inst_dict

        # TODO: Crosstalk specification?
        return

    @property
    def gate_keys(self) -> list:
        return list(self.gate_dict.keys())

    @property
    def instrument_keys(self) -> list:
        return list(self.inst_dict.keys())

    @property
    def output_gate_reps(self) -> list[type[GateRep]]:
        return self._gatereps

    @property
    def output_instrument_reps(self) -> list[type[InstrumentRep]]:
        return self._instreps

    @singledispatchmethod
    def get_reps(
        self,
        circuit: BasePhysicalCircuit,
        gatereps: Sequence[type[GateRep]],
        instreps: Sequence[type[InstrumentRep]],
    ) -> list[OperationRep]:
        """Get list of operator representations that can be applied.

        The default implementation handles any circuit castable to a
        [](api:ListPhysicalCircuit) via a per-label dictionary lookup; a
        separate implementation is registered for
        [](api:STIMPhysicalCircuit) (see the module-level `_get_reps_stim`
        function below) that parses the circuit's STIM text directly.
        Both implementations ignore the `gatereps`/`instreps` arguments,
        since the representation to use is unambiguous from `gate_dict`/
        `inst_dict`'s own values.

        Parameters
        ----------
        circuit:
            Physical circuit to get the representations for

        gatereps:
            Unused; present for interface compatibility with
            [](api:BaseNoiseModel.get_reps).

        instreps:
            Unused; present for interface compatibility with
            [](api:BaseNoiseModel.get_reps).

        Returns
        -------
        list
            List of operation representations for the circuit
        """
        # Get builtin circuit for easy processing (avoiding a redundant
        # copy if it's already the right type).
        if not isinstance(circuit, ListPhysicalCircuit):
            circuit = ListPhysicalCircuit(circuit)

        # Iterate through circuit and pull out representations
        reps = []
        for layer in circuit.circuit:
            for label in layer:
                # Try to look up in gates
                rep = self.gate_dict.get(label, None)

                if rep is None:
                    # Also try to look up just by name
                    generic = self.gate_dict.get(label[0], None)
                    if generic is not None:
                        assert isinstance(generic, GateRep)
                        rep = generic.with_qubits(label[1])

                if rep is None:
                    # Failed, now look up in instruments
                    rep = self.inst_dict.get(label, None)

                if rep is None:
                    # Also try to look up just by name
                    generic = self.inst_dict.get(label[0], None)
                    if generic is not None:
                        assert isinstance(generic, InstrumentRep)
                        rep = generic.with_qubits(label[1])

                assert rep is not None, f"Failed to look up {label}"
                assert isinstance(rep, OperationRep)

                reps.append(rep)
        return reps

    @classmethod
    def _from_decoded_attrs(cls: type[T], attr_dict: Mapping) -> T:
        gate_dict = attr_dict["gate_dict"]
        inst_dict = attr_dict["inst_dict"]
        # New-style files serialize `_gatereps`/`_instreps` entries as bare
        # `type[GateRep]`/`type[InstrumentRep]` classes, which decode
        # directly to the correct class. Old files serialized each entry
        # as a `GateRep`/`InstrumentRep` *enum member*, which decodes to a
        # legacy tag instead (see `GateRep._from_decoded_attrs`) --
        # `upgrade_legacy_gaterep_tag` translates that tag to the
        # corresponding modern class, and passes anything else through
        # unchanged.
        gatereps = [upgrade_legacy_gaterep_tag(v) for v in attr_dict["_gatereps"]]
        instreps = [
            upgrade_legacy_instrumentrep_tag(v) for v in attr_dict["_instreps"]
        ]
        return cls((gate_dict, inst_dict), gatereps, instreps)


def _merge_common_rep(
    command: str,
    qt: tuple,
    generic: OperationRep,
    common: dict[str, OperationRep],
) -> None:
    """Merge a generic (name-only) dict entry into `common[command]`.

    For reps carrying a [](api:StimCircuitPayloadMixin) payload (i.e.
    [](api:StimCircuitGateRep)/[](api:StimCircuitInstrumentRep)), generic
    templates are combined across multiple qubits by *appending* further
    trailing qubit indices to every line for each additional qubit, so a
    template must already reference its own qubit(s) as indices
    ``0..len(qt)-1`` (e.g. `"X 0"` for a single-qubit template, `"CNOT 0 1"`
    for a two-qubit one) before it is used for the first time here. Without
    this, the first qubit's index would silently never be added -- this
    raises a clear error instead of producing a rep that claims to act on a
    qubit its own circuit string never actually references.

    For other rep classes (e.g. [](api:ZBasisProjectionInstrumentRep)),
    there is no per-qubit text to combine -- the same payload applies
    uniformly regardless of how many qubits are merged in (state backends
    apply it in a `for qbit in qubits: ...` loop), so merging is just
    concatenating the qubit tuples.
    """
    prev = common.get(command)

    if not isinstance(generic, StimCircuitPayloadMixin):
        prev_qubits: tuple = prev.qubits if prev is not None else tuple()
        common[command] = generic.with_qubits(prev_qubits + qt)
        return

    if prev is None:
        # First occurrence: the raw template must already reference its own
        # qubit(s) as indices 0..len(qt)-1, and is used exactly as registered
        # -- no further indices are appended for the qubit(s) it already
        # covers. Only later occurrences (below) append further trailing
        # indices for *additional* qubits.
        expected = [str(i) for i in range(len(qt))]
        for line in generic.circuit_str.split("\n"):
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
        common[command] = generic.with_qubits(qt)
        return

    new_lines = [
        line + "".join(f" {len(prev.qubits) + i}" for i in range(len(qt)))
        for line in prev.circuit_str.split("\n")
    ]
    merged = copy.copy(generic)
    merged.circuit_str = "\n".join(new_lines)
    merged.qubits = prev.qubits + qt
    common[command] = merged


def add_command_aliases(d: dict[MemberLabel, Any]) -> None:
    """Add STIM command-alias keys (e.g. `"CNOT"` -> `"CX"`) alongside originals.

    This always *adds* the aliased key alongside the original rather than
    replacing it, so both spellings resolve to the same rep.
    """
    assert STIMPhysicalCircuit is not None
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


if STIMPhysicalCircuit is not None:  # noqa: C901
    # NOTE: mccabe (the `C901` complexity checker) attributes
    # `_get_reps_stim`'s own complexity to this enclosing module-level
    # `if` rather than to the nested function itself, since the function
    # is defined inside a conditional block rather than at module or
    # class scope -- this is a checker quirk of the soft-dependency
    # idiom below (matching `stimcircuit.py`/`stimstate.py`), not a sign
    # `_get_reps_stim` itself needs restructuring.

    # NOTE: `cls` is passed explicitly (rather than relying on
    # `singledispatchmethod.register`'s annotation-auto-detection) because
    # auto-detection inspects the *first* parameter's annotation -- which,
    # for a plain function registered from outside the class body like
    # this, is `self`, not `circuit`. Passing `STIMPhysicalCircuit`
    # explicitly sidesteps that ambiguity entirely.
    @DictNoiseModel.get_reps.register(STIMPhysicalCircuit)
    def _get_reps_stim(
        self: DictNoiseModel,
        circuit: STIMPhysicalCircuit,
        gatereps: Sequence[type[GateRep]],
        instreps: Sequence[type[InstrumentRep]],
    ) -> list[OperationRep]:
        """Get operation representations for a STIM circuit.

        This parses `circuit`'s own STIM text directly (rather than using
        `gate_dict`/`inst_dict`'s pre-structured keys the way the default
        [](api:DictNoiseModel.get_reps) implementation does), since
        [](api:STIMPhysicalCircuit) does not organize its content into
        layers/labels the way [](api:ListPhysicalCircuit) does.

        Unlike the raw dict passed to [](api:DictNoiseModel.__init__),
        which may use any case and any STIM command aliases the user
        chooses, STIM command names in `gate_dict`/`inst_dict` keys are
        looked up here in a case-insensitive, alias-resolved manner (see
        `add_command_aliases`), matching STIM's own case/alias-insensitive
        command parsing.

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
            Unused; present for interface compatibility with
            [](api:BaseNoiseModel.get_reps).

        instreps:
            Unused; present for interface compatibility with
            [](api:BaseNoiseModel.get_reps).

        Returns
        -------
        list[OperationRep]
            List of operation representations describing the circuit's
            operations with noise models applied.
        """

        def promoted_key_and_qubits(k):
            # By convention, STIM commands are looked up case-insensitively
            # (matching STIM's own parsing), so both `gate_dict`/`inst_dict`
            # keys and circuit command names are uppercased before lookup.
            if isinstance(k, str):
                return k.upper(), tuple()
            else:
                return (k[0].upper(), k[1]), k[1]

        gate_dict = {
            promoted_key_and_qubits(k)[0]: v for k, v in self.gate_dict.items()
        }
        inst_dict = {
            promoted_key_and_qubits(k)[0]: v for k, v in self.inst_dict.items()
        }
        add_command_aliases(gate_dict)
        add_command_aliases(inst_dict)

        # Iterate through circuit and pull out representations
        reps = []
        for line in circuit._unroll_repeats().split("\n"):
            entries = line.split()

            if len(entries) == 0:
                # Empty line, but pass it on as a comment so full circuit has same formatting
                reps.append(StimCircuitGateRep(line, tuple()))
                continue

            # Some commands can have parameters
            # Strip those so we can check base command name
            command = entries[0].split("(")[0]

            if command not in circuit._stim_gates:
                # We don't want to just ignore this, pass it on as a dummy rep
                # so that it will be included in the full circuit, but
                # we don't need to handle it here.
                # The empty qubit label will stand for this dummy/comment type rep
                reps.append(StimCircuitGateRep(line, tuple()))

                # We'll warn if this is a noise channel, just FYI since we are
                # adding more noise
                if command in circuit._stim_noise_channels:
                    warnings.warn(
                        f"Noise channel '{line}' detected, but DictNoiseModel is also adding noise"
                    )
            else:
                # This is a gate!
                # First check if measure had noise applied and warn
                if (
                    command in circuit._stim_measure_reset_gates
                    and "(" in entries[0]
                ):
                    warnings.warn(
                        f"Measure noise '{entries[0]}' detected, but DictNoiseModel is also adding noise"
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
                common: dict[str, OperationRep] = {}
                for qt in qubit_tuples:
                    is_common = False
                    label = (command.upper(), qt)

                    # Try to look up in gates
                    rep = gate_dict.get(label, None)

                    if rep is None:
                        # If that failed, check for generic name only
                        generic = gate_dict.get(command.upper(), None)
                        if generic is not None:
                            assert isinstance(generic, GateRep)
                            is_common = True
                            _merge_common_rep(command, qt, generic, common)
                            rep = common[command]

                    if rep is None:
                        # Failed, now look up in instruments
                        rep = inst_dict.get(label, None)

                    if rep is None:
                        # If that failed, check for generic name only
                        generic = inst_dict.get(command.upper(), None)
                        if generic is not None:
                            assert isinstance(generic, InstrumentRep)
                            is_common = True
                            _merge_common_rep(command, qt, generic, common)
                            rep = common[command]

                    assert rep is not None, f"Failed to look up {label}"
                    assert isinstance(rep, OperationRep)

                    if not is_common:
                        reps.append(rep)

                # Add common commands to the rep
                reps.extend(common.values())

        return reps
