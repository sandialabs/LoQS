#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
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
from typing import (
    ClassVar,
    Literal,
    TypeAlias,
    TypeVar,
    TYPE_CHECKING,
    Any,
    cast,
)

from loqs.backends.circuit import BasePhysicalCircuit, ListPhysicalCircuit
from loqs.backends.model import BaseNoiseModel
from loqs.backends.model.basemodel import first_successful_rep
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    KrausGateRep,
    OperationRep,
    ProbabilisticStimGateRep,
    PTMGateRep,
    QSimSuperopGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    StimCircuitPayloadMixin,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
    convert as convert_rep,
    is_rep_compatible,
)
from loqs.backends.reps.conversion import reachable_reps
from loqs.backends.reps.legacy import (
    upgrade_legacy_gaterep_tag,
    upgrade_legacy_instrumentrep_tag,
)

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


def _lookup(
    label: tuple, gate_dict: Mapping, inst_dict: Mapping
) -> tuple[Any, bool]:
    """Find `label`'s dict entry: exact gate key, gate name, exact
    instrument key, then instrument name. Returns `(entry, generic)`."""
    for d in (gate_dict, inst_dict):
        for key in (label, label[0]):
            entry = d.get(key)
            if entry is not None:
                return entry, key is not label
    raise KeyError(f"Failed to look up {label}")


class DictNoiseModel(BaseNoiseModel):
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
        gate_dict: Mapping | None = None,
        inst_dict: Mapping | None = None,
        gatereps: Sequence[type[GateRep]] = (QSimSuperopGateRep,),
        instreps: Sequence[type[InstrumentRep]] = (
            ZBasisProjectionInstrumentRep,
        ),
        gaterep_array_cast_rep: (
            type[UnitaryGateRep] | type[PTMGateRep] | type[QSimSuperopGateRep]
        ) = QSimSuperopGateRep,
        instrep_cast_reset: Literal[0, 1, None] = None,
        instrep_cast_include_outcomes: bool = True,
    ) -> None:
        """Initialize a generic gate dict model.

        Parameters
        ----------
        gate_dict:
            See [](api:gate_dict). Defaults to `None`, which uses an
            empty `dict`.

        inst_dict:
            See [](api:inst_dict). Defaults to `None`, which uses an
            empty `dict`.

        gatereps:
            [](api:GateRep) classes the dict entries may hold.
            [](api:DictNoiseModel.get_reps) converts them to other
            classes when a caller asks for those.

        instreps:
            [](api:InstrumentRep) classes the dict entries may hold, which
            `get_reps` converts likewise.

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
            [](api:OutcomeOperationDictInstrumentRep) values are
            being cast up from raw values, this will be used as the
            `include_outcome` field, indicating whether outcomes should be
            kept (`True`, default) or not (`False`).
        """

        # NOTE: We set self.gate_dict and self.inst_dict at the end of this
        #  function. The next two variables are like gate_dict and inst_dict,
        #  but have more lax types.
        gate_dict = dict(gate_dict) if gate_dict is not None else {}
        inst_dict = dict(inst_dict) if inst_dict is not None else {}

        self._gatereps = list(gatereps)
        self._instreps = list(instreps)
        self._converted: dict = {}

        def convert_to_gaterep(gr, qubits) -> GateRep:
            if isinstance(gr, GateRep):
                assert any(
                    isinstance(gr, cls) for cls in gatereps
                ), f"Provided {gr} but not provided gatereps"
                return gr
            if isinstance(gr, np.ndarray):
                # matrix for dense rep; None (unknown qubits) means "not attached yet".
                return gaterep_array_cast_rep(
                    gr, () if qubits is None else qubits
                )
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
                assert OutcomeOperationDictInstrumentRep in instreps, (
                    "Detected dict for a outcome-operation instrument, but "
                    + "OutcomeOperationDictInstrumentRep not passed as a valid instrument rep"
                )
                outcome_ops = {
                    k: convert_to_gaterep(v, qubits) for k, v in ir.items()
                }
                return OutcomeOperationDictInstrumentRep(
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

    @classmethod
    def from_model(
        cls: type[T],
        model: BaseNoiseModel,
        gatereps: Sequence[type[GateRep]] | None = None,
        instreps: Sequence[type[InstrumentRep]] | None = None,
        **kwargs,
    ) -> T:
        """Build a [](api:DictNoiseModel) by converting an existing model.

        Unlike [](api:DictNoiseModel.__init__) (which takes already
        laid-out `gate_dict`/`inst_dict` mappings), this converts between
        two already fully-built model objects: either copying another
        [](api:DictNoiseModel)'s dicts, or building fresh ones from a
        `PyGSTiNoiseModel` (matched by `type(model).__name__` rather than
        `isinstance`, so this works without hard-requiring `pygsti` to be
        importable). Keys are `(name, tuple(qubits))`, using the
        `PyGSTiNoiseModel`'s aliased qubit labels.

        Parameters
        ----------
        model:
            The model to convert from. A `PyGSTiNoiseModel` that uses time
            dependence raises `NotImplementedError`.

        gatereps:
            [](api:GateRep) classes the new model's entries may hold, and
            the classes requested from a `PyGSTiNoiseModel`. These must
            describe the entries being copied. `None` means the source's
            own declared classes for a [](api:DictNoiseModel), and
            [](api:QSimSuperopGateRep) for a `PyGSTiNoiseModel`.

        instreps:
            [](api:InstrumentRep) classes likewise. `None` means the
            source's own declared classes for a [](api:DictNoiseModel), and
            [](api:ZBasisProjectionInstrumentRep) for a `PyGSTiNoiseModel`.

        **kwargs:
            Forwarded to [](api:DictNoiseModel.__init__).

        Returns
        -------
            A [](api:DictNoiseModel) equivalent to `model`.
        """
        if isinstance(model, DictNoiseModel):
            return cls(
                model.gate_dict.copy(),
                model.inst_dict.copy(),
                gatereps=model._gatereps if gatereps is None else gatereps,
                instreps=model._instreps if instreps is None else instreps,
                **kwargs,
            )
        if type(model).__name__ != "PyGSTiNoiseModel":
            raise TypeError("Can only convert from other NoiseModels")
        if getattr(model, "use_time_dependence", False):
            raise NotImplementedError(
                "Can't convert a time-dependent model to a DictNoiseModel"
            )

        gatereps = (QSimSuperopGateRep,) if gatereps is None else gatereps
        instreps = (
            (ZBasisProjectionInstrumentRep,) if instreps is None else instreps
        )
        gate_dict: dict = {}
        inst_dict: dict = {}
        for keys, out in (
            (model.gate_keys, gate_dict),
            (model.instrument_keys, inst_dict),
        ):
            for name, qubits in keys:
                label = (name, tuple(qubits))
                out[label] = model.get_reps(
                    ListPhysicalCircuit([[label]]), gatereps, instreps
                )[0]
        return cls(
            gate_dict,
            inst_dict,
            gatereps=gatereps,
            instreps=instreps,
            **kwargs,
        )

    @property
    def gate_keys(self) -> list:
        return list(self.gate_dict.keys())

    @property
    def instrument_keys(self) -> list:
        return list(self.inst_dict.keys())

    @property
    def output_gate_reps(self) -> list[type[GateRep]]:
        reps = list(reachable_reps(tuple(self._gatereps)))
        if StimCircuitGateRep not in reps:
            # Annotation and noise-channel lines are built as this type
            reps.append(StimCircuitGateRep)
        return cast(list[type[GateRep]], reps)

    @property
    def output_instrument_reps(self) -> list[type[InstrumentRep]]:
        return cast(
            list[type[InstrumentRep]],
            list(reachable_reps(tuple(self._instreps))),
        )

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
        A rep not of a requested type is converted to the first type that
        works, in order; one that converts to none raises
        [](api:RepConstructionError).

        Parameters
        ----------
        circuit:
            Physical circuit to get the representations for

        gatereps:
            Allowed output [](api:GateRep) classes.

        instreps:
            Allowed output [](api:InstrumentRep) classes.

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
        reps: list[OperationRep] = []
        for layer in circuit.circuit:
            for label in layer:
                entry, generic = _lookup(label, self.gate_dict, self.inst_dict)
                reps.append(
                    self._as_requested(
                        entry, generic, label[1], gatereps, instreps
                    )
                )
        return reps

    def _as_requested(
        self,
        entry: OperationRep,
        generic: bool,
        qubits: tuple,
        gatereps: Sequence[type[GateRep]],
        instreps: Sequence[type[InstrumentRep]],
    ) -> OperationRep:
        """Return `entry` (retargeted to `qubits` if generic) as the first
        requested type that works, caching conversions per entry object."""
        types = gatereps if isinstance(entry, GateRep) else instreps
        if isinstance(entry, tuple(types)):
            return entry.with_qubit_labels(qubits) if generic else entry
        return first_successful_rep(
            self._converted,
            (entry, qubits),
            types,
            lambda: entry.with_qubit_labels(qubits) if generic else entry,
            convert_rep,
            kind="gate rep" if types is gatereps else "instrument rep",
        )

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
        gatereps = cast(
            list[type[GateRep]],
            [upgrade_legacy_gaterep_tag(v) for v in attr_dict["_gatereps"]],
        )
        instreps = cast(
            list[type[InstrumentRep]],
            [
                upgrade_legacy_instrumentrep_tag(v)
                for v in attr_dict["_instreps"]
            ],
        )
        return cls(gate_dict, inst_dict, gatereps, instreps)


def build_legacy_stim_dict_model(
    model_or_dicts: "DictNoiseModel | tuple[Mapping, Mapping]",
    gatereps: Sequence[type[GateRep]] | None = None,
    instreps: Sequence[type[InstrumentRep]] | None = None,
    gaterep_array_cast_rep: (
        type[UnitaryGateRep]
        | type[PTMGateRep]
        | type[QSimSuperopGateRep]
        | None
    ) = None,
    instrep_cast_reset: Literal[0, 1, None] = None,
    instrep_cast_include_outcomes: bool = True,
) -> "DictNoiseModel":
    """Build a [](api:DictNoiseModel) from a pre-1.2 `STIMDictNoiseModel(...)`
    call's arguments -- the `build=` callable behind that class's legacy
    construction shim (see `loqs.backends.model.__init__`).

    `STIMDictNoiseModel` was removed in v1.2 because its one real feature
    (case/alias-insensitive STIM command lookup) is now built directly
    into [](api:DictNoiseModel.get_reps)'s own STIM handling, applied
    uniformly regardless of how `gate_dict`/`inst_dict` were built -- so a
    plain [](api:DictNoiseModel) needs no separate STIM-aware subclass.
    Positional-argument order and meaning otherwise match the old
    constructor exactly, except `model_or_dicts` accepting a
    [](api:DictNoiseModel) instance to copy in place of the old class's
    own instances (never possible to receive one of those anymore).
    """
    if isinstance(model_or_dicts, DictNoiseModel):
        gate_dict = dict(model_or_dicts.gate_dict)
        inst_dict = dict(model_or_dicts.inst_dict)
    elif isinstance(model_or_dicts, tuple) and len(model_or_dicts) == 2:
        gate_dict = dict(model_or_dicts[0])
        inst_dict = dict(model_or_dicts[1])
    else:
        raise TypeError(
            "STIMDictNoiseModel's first argument must be an existing "
            "noise model to copy or a 2-tuple of (gate_dict, inst_dict)."
        )

    kwargs: dict[str, Any] = dict(
        gate_dict=gate_dict,
        inst_dict=inst_dict,
        gatereps=(
            list(gatereps)
            if gatereps is not None
            else [StimCircuitGateRep, ProbabilisticStimGateRep]
        ),
        instreps=(
            list(instreps)
            if instreps is not None
            else [StimCircuitInstrumentRep]
        ),
        instrep_cast_reset=instrep_cast_reset,
        instrep_cast_include_outcomes=instrep_cast_include_outcomes,
    )
    if gaterep_array_cast_rep is not None:
        kwargs["gaterep_array_cast_rep"] = gaterep_array_cast_rep
    return DictNoiseModel(**kwargs)


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
    """
    prev = common.get(command)
    assert isinstance(generic, StimCircuitPayloadMixin)

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
        common[command] = generic.with_qubit_labels(qt)
        return

    assert isinstance(prev, StimCircuitPayloadMixin)
    new_lines = [
        line
        + "".join(f" {len(prev.qubit_labels) + i}" for i in range(len(qt)))
        for line in prev.circuit_str.split("\n")
    ]
    merged = copy.copy(generic)
    merged.circuit_str = "\n".join(new_lines)
    merged.qubit_labels = prev.qubit_labels + qt
    common[command] = merged


def _stim_group_reps(
    model: DictNoiseModel,
    command: str,
    groups: list[tuple],
    dicts: tuple[Mapping, Mapping],
    gatereps: Sequence[type[GateRep]],
    instreps: Sequence[type[InstrumentRep]],
) -> list[OperationRep]:
    """One rep per target group of a STIM gate line, in target order.

    Generic STIM-text reps already of a requested type merge into one rep
    over disjoint groups; any other rep, and any group sharing a qubit with
    the pending merged rep, flushes it first.
    """
    reps: list[OperationRep] = []
    common: dict[str, OperationRep] = {}
    for qt in groups:
        entry, generic = _lookup((command, qt), *dicts)
        # A group holds one or two labels, so this checks each of them
        if generic and not (
            isinstance(qt[0], str) and isinstance(qt[-1], str)
        ):
            stale = (command, tuple(str(q) for q in qt))
            if stale in dicts[0] or stale in dicts[1]:
                raise KeyError(
                    f"Key {stale} matches no label of type "
                    f"{[type(q).__name__ for q in qt]}; use the label itself"
                )
        mergeable = (
            generic
            and isinstance(entry, StimCircuitPayloadMixin)
            and isinstance(
                entry,
                tuple(gatereps if isinstance(entry, GateRep) else instreps),
            )
        )
        if command in common and (
            not mergeable
            or not set(qt).isdisjoint(common[command].qubit_labels)
        ):
            reps.append(common.pop(command))
        if mergeable:
            _merge_common_rep(command, qt, entry, common)
        else:
            reps.append(
                model._as_requested(entry, generic, qt, gatereps, instreps)
            )
    reps.extend(common.values())
    return reps


def add_command_aliases(d: dict[MemberLabel, Any]) -> None:
    """Add STIM command-alias keys (e.g. `"CNOT"` -> `"CX"`) alongside originals.

    This always *adds* the aliased key alongside the original rather than
    replacing it, so both spellings resolve to the same rep.
    """
    assert STIMPhysicalCircuit is not None
    aliases = STIMPhysicalCircuit.stim_command_aliases

    need_aliasing: list[MemberLabel] = []
    for k in d:
        if isinstance(k, str) and k in aliases:
            need_aliasing.append(k)
        if isinstance(k, tuple) and k[0] in aliases:
            need_aliasing.append(k)

    for k in need_aliasing:
        aliased_k: MemberLabel
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
    # A known typeshed limitation: singledispatchmethod.__get__'s stub doesn't
    # expose .register when accessed on the class.
    @DictNoiseModel.get_reps.register(STIMPhysicalCircuit)  # type: ignore[attr-defined]
    def _get_reps_stim(
        self: DictNoiseModel,
        circuit: STIMPhysicalCircuit,
        gatereps: Sequence[type[GateRep]],
        instreps: Sequence[type[InstrumentRep]],
    ) -> list[OperationRep]:
        """Get operation representations for a STIM circuit.

        This reads `circuit`'s own STIM text through
        `STIMPhysicalCircuit._parsed_lines`, which does the parsing: line
        kinds, target mapping and target groups (rather than using
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
        - Mapping targets to qubit labels, keeping the labels' types
        - One rep per target group of a gate line (pairs for two-qubit
          gates and `MXX`/`MYY`/`MZZ`), in target order, from the gate and
          instrument dictionaries
        - Merging generic STIM-text reps over disjoint groups
        - Passing annotations and noise-channel lines on as STIM text, only
          when `StimCircuitGateRep` is requested; a noise channel is
          refused otherwise and warned about when applied
        - Refusing (`NotImplementedError`) any instruction or target it
          can't map, such as tagged gates or `rec[-1]` targets

        Parameters
        ----------
        circuit:
            The STIM circuit to process.

        gatereps:
            Allowed output [](api:GateRep) classes; a rep not of one is
            converted to the first that works, else
            [](api:RepConstructionError) is raised.

        instreps:
            Allowed output [](api:InstrumentRep) classes, used likewise.

        Returns
        -------
        list[OperationRep]
            List of operation representations describing the circuit's
            operations with noise models applied.
        """

        def promoted_key(k):
            # By convention, STIM commands are looked up case-insensitively
            # (matching STIM's own parsing), so both `gate_dict`/`inst_dict`
            # keys and circuit command names are uppercased before lookup.
            return k.upper() if isinstance(k, str) else (k[0].upper(), k[1])

        gate_dict = {promoted_key(k): v for k, v in self.gate_dict.items()}
        inst_dict = {promoted_key(k): v for k, v in self.inst_dict.items()}
        add_command_aliases(gate_dict)
        add_command_aliases(inst_dict)

        # Iterate through circuit and pull out representations
        reps: list[OperationRep] = []
        want_text = is_rep_compatible(StimCircuitGateRep, gatereps)
        for kind, line, name, args, groups in circuit._parsed_lines():
            if kind == "annotation":
                # Annotations act on no qubit; the empty label stands for
                # them and keeps the circuit's format
                if want_text:
                    reps.append(StimCircuitGateRep(line, tuple()))
            elif kind == "noise":
                if not want_text:
                    raise NotImplementedError(
                        f"Noise channel '{line}' needs a state that takes "
                        "StimCircuitGateRep"
                    )
                warnings.warn(
                    f"Noise channel '{line}' detected, but DictNoiseModel is also adding noise"
                )
                indices = map(str, range(len(groups[0])))
                reps.append(
                    StimCircuitGateRep(
                        " ".join([f"{name}{args}", *indices]), groups[0]
                    )
                )
            else:
                if name in circuit._stim_measure_reset_gates and args:
                    warnings.warn(
                        f"Measure noise '{name}{args}' detected, but the "
                        "instruction's noise argument is ignored and the "
                        "model's entry is used instead"
                    )
                reps.extend(
                    _stim_group_reps(
                        self,
                        name,
                        groups,
                        (gate_dict, inst_dict),
                        gatereps,
                        instreps,
                    )
                )

        return reps
