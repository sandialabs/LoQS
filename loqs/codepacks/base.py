#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""The base class of all codepacks, and its shared ideal noise model."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from loqs.backends.circuit import BasePhysicalCircuit
from loqs.backends.model.basemodel import BaseNoiseModel
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.reps import (
    GateRep,
    InstrumentRep,
    QSimSuperopGateRep,
    STANDARD_GATE_UNITARIES,
    UnitaryGateRep,
    ZBasisProjectionInstrumentRep,
    convert as convert_rep,
)
from loqs.core import Instruction, QECCode
from loqs.core.instructions import InstructionLabel

if TYPE_CHECKING:
    from loqs.backends.model.pygstimodel import PyGSTiNoiseModel
else:
    try:
        from loqs.backends import PyGSTiNoiseModel
    except ImportError:
        PyGSTiNoiseModel = Any  # type: ignore

_STANDARD_GATE_NAMES = {
    "Gxpi": "X",
    "Gypi": "Y",
    "Gzpi": "Z",
    "Gzpi2": "S",
    "Gzmpi2": "S_DAG",
    "Gh": "H",
    "Gcnot": "CX",
    "Gcphase": "CZ",
    "Gi": "I",
}
"""pyGSTi-style gate names, mapped to keys of `STANDARD_GATE_UNITARIES`."""


def walk_qec_code_circuits(code: QECCode) -> list[BasePhysicalCircuit]:
    """Recursively walk QEC code instructions and collect circuit leaves.

    Walks through every Instruction in the code, resolving composite and
    repeat-until-success instructions (by object or key) down to
    physical-circuit leaves. Returns each reachable circuit once.
    """
    visited: set[int] = set()
    circuits: list[BasePhysicalCircuit] = []

    def _walk(inst: Instruction) -> None:
        inst_id = id(inst)
        if inst_id in visited:
            return
        visited.add(inst_id)

        data = inst.data
        if "instructions" in data:
            # Composite-like instruction (Composite or Repeat-until-success)
            for entry in data["instructions"]:
                label = InstructionLabel.from_raw(entry)
                if "patch_label" in label:
                    raise ValueError(
                        "Patch-level instruction resolution is out of scope for "
                        "a single QECCode's static instruction tree"
                    )
                if "instruction" in label:
                    instr_entry = label["instruction"]
                    if isinstance(instr_entry, Instruction):
                        _walk(instr_entry)
                    elif isinstance(instr_entry, str):
                        resolved = code.instructions[instr_entry]
                        _walk(resolved)
        elif "circuit" in data:
            # Circuit leaf instruction
            circuits.append(data["circuit"])

    # Walk all top-level instructions
    for inst in code.instructions.values():
        _walk(inst)

    return circuits


def walk_qec_code_gate_names(code: QECCode) -> set[str]:
    """Recursively walk QEC code instructions and collect gate names.

    The union of the `used_gate_names` of every circuit leaf from
    [](api:walk_qec_code_circuits). Returns an undifferentiated flat set
    mixing gate names (e.g. "Gxpi") with instrument names (e.g. "Imrz").
    """
    gate_names: set[str] = set()
    for circuit in walk_qec_code_circuits(code):
        gate_names.update(circuit.used_gate_names)
    return gate_names


class BaseCodepack(ABC):
    """Base class of all codepacks.

    A codepack takes its configuration once, in its own `__init__`, and
    stores it on the instance. Configuration must reach instructions
    through `Instruction.data`, never through a closure over `self`:
    instructions are serialized as source and re-run in a fresh scope.

    Subclasses implement [](api:BaseCodepack.create_qec_code). They may
    set `_nonstd_unitaries` (unitaries of gate names that are not standard)
    and `_idle_gate_names` (names of one-qubit identity gates), on the class
    or on the instance when the names depend on the configuration.
    """

    _nonstd_unitaries: Mapping[str, NDArray] = {}
    _idle_gate_names: Sequence[str] = ("Gi1Q", "Gi2Q", "GiMCM")

    @abstractmethod
    def create_qec_code(self) -> QECCode:
        """Create the [](api:QECCode) of this codepack."""

    def _gate_unitary(self, name: str) -> NDArray | None:
        if name in self._nonstd_unitaries:
            return np.asarray(self._nonstd_unitaries[name])
        if name in self._idle_gate_names:
            return STANDARD_GATE_UNITARIES["I"]
        if name in _STANDARD_GATE_NAMES:
            return STANDARD_GATE_UNITARIES[_STANDARD_GATE_NAMES[name]]
        return None

    def create_ideal_model(  # noqa: C901
        self,
        qubits: Sequence[str],
        model_backend: type[BaseNoiseModel] = PyGSTiNoiseModel,
        gaterep: type[GateRep] = QSimSuperopGateRep,
        instrep: type[InstrumentRep] = ZBasisProjectionInstrumentRep,
    ) -> BaseNoiseModel:
        """Create an ideal (noiseless) model for this codepack's code.

        The model holds exactly the gates and instruments that the code's
        instructions use. A pyGSTi model of up to 31 qubits is a
        density-matrix model, so its gates can be replaced by noisy pyGSTi
        operations; above 31 qubits, where pyGSTi can't build one, it is a
        state-vector model, which holds only unitary gates.

        Parameters
        ----------
        qubits : Sequence[str]
            Qubit labels of the register.

        model_backend : type[BaseNoiseModel], optional
            [](api:PyGSTiNoiseModel) (default) or [](api:DictNoiseModel).

        gaterep : type[GateRep], optional
            Gate representation class. Default is [](api:QSimSuperopGateRep).

        instrep : type[InstrumentRep], optional
            Instrument representation class. Default is
            [](api:ZBasisProjectionInstrumentRep).

        Returns
        -------
        BaseNoiseModel
            The ideal model.

        Raises
        ------
        ValueError
            If the code uses an instrument other than `Imrz` or a gate
            with no known unitary, or for an unsupported pyGSTi option.
        """
        walked = walk_qec_code_gate_names(self.create_qec_code())
        gates = {n for n in walked if n.startswith("G")}
        has_imrz = "Imrz" in walked
        unsupported = sorted(
            n
            for n in walked
            if (n.startswith("G") and self._gate_unitary(n) is None)
            or (not n.startswith("G") and n != "Imrz")
        )
        if unsupported:
            raise ValueError(
                f"Cannot build an ideal model for gates/instruments {unsupported}"
            )
        unitaries = {n: self._gate_unitary(n) for n in sorted(gates)}

        if model_backend is PyGSTiNoiseModel:
            try:
                import pygsti
            except ImportError:
                raise ImportError(
                    "pyGSTi not found, cannot construct pyGSTi noise model"
                )
            # pyGSTi's densitymx embedding overflows a C long above 31 qubits
            evotype = "densitymx" if len(qubits) <= 31 else "statevec"
            if instrep is not ZBasisProjectionInstrumentRep:
                raise ValueError(
                    "The pyGSTi ideal model supports only "
                    "ZBasisProjectionInstrumentRep"
                )
            model_qubits = [f"Q{i}" for i in range(len(qubits))]
            nonstd = {
                n: U
                for n, U in unitaries.items()
                if n in self._nonstd_unitaries or n in self._idle_gate_names
            }
            # No state prep or POVM: pyGSTi would build dense 2^n vectors
            pspec = pygsti.processors.QubitProcessorSpec(
                len(model_qubits),
                gate_names=list(unitaries),
                qubit_labels=model_qubits,
                nonstd_gate_unitaries=nonstd,
                availability={k: "all-permutations" for k in unitaries},
                prep_names=(),
                povm_names=(),
            )
            inner = pygsti.models.create_crosstalk_free_model(
                pspec, evotype=evotype
            )
            model = PyGSTiNoiseModel(inner, qubits)
            if gaterep not in model.output_gate_reps:
                raise ValueError(
                    f"{gaterep.__name__} is not an output gate rep of the "
                    "pyGSTi model"
                )
            return model
        elif model_backend is DictNoiseModel:
            gate_dict = {}
            for name, U in unitaries.items():
                assert U is not None
                num_qubits = int(np.log2(U.shape[0]))
                for qs in itertools.permutations(qubits, r=num_qubits):
                    gate_dict[(name, qs)] = convert_rep(
                        UnitaryGateRep(U, qs), gaterep, qs
                    )
            # (0, True): reset to 0 and record the outcome
            inst_dict = (
                {("Imrz", (q,)): (0, True) for q in qubits} if has_imrz else {}
            )
            return DictNoiseModel(
                gate_dict, inst_dict, gatereps=[gaterep], instreps=[instrep]
            )
        elif isinstance(model_backend, type) and issubclass(
            model_backend, BaseNoiseModel
        ):
            raise NotImplementedError(
                "Cannot generate ideal model for this backend"
            )
        else:
            raise ValueError("Must pass a noise model class")
