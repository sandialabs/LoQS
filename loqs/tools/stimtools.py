#####################################################################################################################
# Logical Qubit Simulator (LoQS) v. 1.3                                                                           #
# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).                                #
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software. #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except                  #
# in compliance with the License.  You may obtain a copy of the License at                                          #
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root LoQS directory.                     #
#####################################################################################################################

"""Helpers for running STIM circuits in LoQS.

[](api:build_stim_model) builds the ideal model for a STIM circuit's gates,
[](api:stim_record_from_outcomes) rebuilds STIM's measurement record,
detector events and observable flips from the outcomes of a LoQS run, and
[](api:build_stim_decoder_instruction) decodes them with PyMatching.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import itertools
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
import stim

from loqs.backends.circuit.stimcircuit import STIMPhysicalCircuit
from loqs.backends.model.dictmodel import DictNoiseModel
from loqs.backends.reps import (
    GateRep,
    KrausGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
    convert as convert_rep,
)
from loqs.backends.state.stimstate import STIMQuantumState
from loqs.core import Frame, Instruction, ProgramResults

# Per instrument name: (basis, measures, resets)
_INSTRUMENTS = {
    "R": ("Z", False, True),
    "RX": ("X", False, True),
    "RY": ("Y", False, True),
    "M": ("Z", True, False),
    "MX": ("X", True, False),
    "MY": ("Y", True, False),
    "MR": ("Z", True, True),
    "MRX": ("X", True, True),
    "MRY": ("Y", True, True),
}


class StimRecord(NamedTuple):
    """STIM's view of a set of runs, each array shaped `(shots, n)`."""

    measurements: NDArray
    detectors: NDArray
    observables: NDArray


def _unitary(name: str) -> NDArray:
    # STIM returns single precision; the polar factor restores unitarity
    tableau = stim.Tableau.from_named_gate(name)
    matrix = tableau.to_unitary_matrix(endian="big").astype(complex)
    u, _, vh = np.linalg.svd(matrix)
    return u @ vh


def _paulis(n: int) -> list[NDArray]:
    singles = [np.eye(2)] + [_unitary(p) for p in "XYZ"]
    return [
        functools.reduce(np.kron, ps)
        for ps in itertools.product(singles, repeat=n)
    ]


def _channel(
    unitary: NDArray,
    errors: list[tuple[NDArray, float]],
    after: bool,
    qubits: tuple,
) -> GateRep:
    """A unitary with a Pauli error channel applied after (or before) it."""
    errors = [(e, w) for e, w in errors if w > 0]
    if all(np.allclose(e, np.eye(len(e))) for e, _ in errors):
        return UnitaryGateRep(unitary, qubits)
    kraus = [
        (np.sqrt(w) * (e @ unitary if after else unitary @ e), float(w))
        for e, w in errors
    ]
    return KrausGateRep(kraus, qubits)


def _flip(basis: str, p: float) -> list[tuple[NDArray, float]]:
    flip = _unitary("Z" if basis == "X" else "X")
    return [(np.eye(2), 1 - p), (flip, p)]


def build_stim_model(
    circuit: STIMPhysicalCircuit | stim.Circuit,
    gaterep: type[GateRep] = StimCircuitGateRep,
    after_clifford_depolarization: float = 0.0,
    after_reset_flip_probability: float = 0.0,
    before_measure_flip_probability: float = 0.0,
) -> DictNoiseModel:
    """Build the model for the gates a STIM circuit uses.

    The model has a generic entry for each unitary gate, reset and
    measurement in the circuit, plus `X`, `Y` and `Z` so error injection
    works. The three probabilities place Pauli noise as
    `stim.Circuit.generated` does: depolarizing noise after each gate
    except `I`, a flip after each reset, and a flip before each measurement
    (and after the reset in `MR`-type measurements). The flip is X-type for
    the Z and Y bases and Z-type for the X basis.

    Parameters
    ----------
    circuit:
        The circuit whose gate set the model covers.

    gaterep:
        [](api:StimCircuitGateRep) (default) builds STIM text for the STIM
        state. Any other [](api:GateRep) class builds dense reps for a dense
        state; with noise, use [](api:KrausGateRep) or
        [](api:QSimSuperopGateRep), since [](api:UnitaryGateRep) can't
        hold it.

    after_clifford_depolarization:
        Depolarizing strength after each gate except `I`.

    after_reset_flip_probability:
        Flip probability after each reset.

    before_measure_flip_probability:
        Flip probability before each measurement.

    A pass-through line (`MPP`, or a gate controlled by a `rec` bit) gets
    no model entry and no model noise: its own arguments and the circuit's
    noise lines pass through as written.

    Returns
    -------
    DictNoiseModel
        The model.

    Raises
    ------
    ValueError
        If a probability is outside `[0, 1]`.

    NotImplementedError
        If the circuit has an instruction other than a unitary gate, reset,
        measurement, annotation, noise channel or `MPP`, or a gate with
        Pauli targets; or if `before_measure_flip_probability` is positive
        and the circuit has an `MPP`.
    """
    pd = after_clifford_depolarization
    pr = after_reset_flip_probability
    pm = before_measure_flip_probability
    if any(not 0 <= p <= 1 for p in (pd, pr, pm)):
        raise ValueError(
            f"Probabilities must be in [0, 1], got {(pd, pr, pm)}"
        )
    if isinstance(circuit, STIMPhysicalCircuit):
        circuit = circuit.circuit

    gates, instruments = ["X", "Y", "Z"], []
    for name in dict.fromkeys(i.name for i in circuit.flattened()):
        data = stim.gate_data(name)
        if name == "MPP":
            if pm > 0:
                raise NotImplementedError(
                    "build_stim_model can't place measurement noise on MPP"
                )
        elif name in _INSTRUMENTS:
            instruments.append(name)
        elif data.is_unitary and not data.takes_pauli_targets:
            gates.append(name)
        elif data.is_unitary or data.produces_measurements or data.is_reset:
            raise NotImplementedError(
                f"STIM instruction {name} can't be built into a model"
            )

    stim_text = gaterep is StimCircuitGateRep
    gate_dict: dict = {}
    for name in dict.fromkeys(gates):
        unitary = _unitary(name)
        n = int(np.log2(unitary.shape[0]))
        qubits = (0, 1)[:n]
        noisy = pd > 0 and name != "I"
        if stim_text:
            targets = " ".join(map(str, qubits))
            text = f"{name} {targets}"
            if noisy:
                text += f"\nDEPOLARIZE{n}({pd}) {targets}"
            gate_dict[name] = StimCircuitGateRep(text, ())
        else:
            errors = [(e, pd / (4**n - 1)) for e in _paulis(n)[1:]]
            errors += [(np.eye(2**n), 1 - pd)]
            rep = _channel(unitary, errors if noisy else [], True, qubits)
            gate_dict[name] = convert_rep(rep, gaterep, qubits)

    inst_dict: dict = {}
    for name in instruments:
        basis, measures, resets = _INSTRUMENTS[name]
        p_before = pm if measures else 0.0
        p_after = pr if resets else 0.0
        if stim_text:
            err = "Z_ERROR" if basis == "X" else "X_ERROR"
            lines = [f"{err}({p_before}) 0"] * bool(p_before)
            lines.append(f"{name} 0")
            lines += [f"{err}({p_after}) 0"] * bool(p_after)
            inst_dict[name] = StimCircuitInstrumentRep("\n".join(lines), ())
            continue
        reset: Literal[0, None] = 0 if resets else None
        if basis == "Z" and not (p_before or p_after):
            inst_dict[name] = ZBasisProjectionInstrumentRep(
                reset, measures, (0,)
            )
            continue
        to_z = {
            "Z": np.eye(2),
            "X": _unitary("H"),
            "Y": _unitary("H") @ _unitary("S_DAG"),
        }[basis]
        pre = _channel(to_z, _flip(basis, p_before), False, (0,))
        post = _channel(to_z.conj().T, _flip(basis, p_after), True, (0,))
        inst_dict[name] = ZBasisPrePostInstrumentRep(
            reset,
            measures,
            convert_rep(pre, gaterep, (0,)),
            convert_rep(post, gaterep, (0,)),
            (0,),
        )

    if stim_text:
        gatereps: list = [StimCircuitGateRep]
        instreps: list = [StimCircuitInstrumentRep]
    else:
        gatereps = [gaterep]
        instreps = [ZBasisProjectionInstrumentRep, ZBasisPrePostInstrumentRep]
    return DictNoiseModel(
        gate_dict, inst_dict, gatereps=gatereps, instreps=instreps
    )


@functools.lru_cache(maxsize=32)
def _label_order(
    circuit_str: str, labels: tuple, unlabeled: bool = False
) -> tuple:
    """The qubit label behind each bit of the measurement record.

    With `unlabeled`, each bit an `MPP` writes has label `None`; any other
    instruction that writes bits without a qubit still raises.
    """
    order: list = []
    for inst in stim.Circuit(circuit_str).flattened():
        # The instructions the STIM state files under a qubit label
        if inst.name in STIMQuantumState._recorded_measurements:
            order += [labels[t.value] for t in inst.targets_copy()]
        elif unlabeled and inst.name == "MPP":
            order += [None] * inst.num_measurements
        elif stim.gate_data(inst.name).produces_measurements:
            raise NotImplementedError(
                f"{inst.name} writes measurement record bits that LoQS "
                "can't attribute to qubits"
            )
    return tuple(order)


@functools.lru_cache(maxsize=32)
def _m2d_converter(circuit_str: str) -> Any:
    return stim.Circuit(circuit_str).compile_m2d_converter()


def _shot_record(shot: Any, order: tuple) -> NDArray:
    """One shot's measurement record, from its per-label outcomes."""
    if isinstance(shot, Mapping):
        shot = [shot]
    outcomes: dict = {}
    for frame in shot:
        for label, vals in (frame or {}).items():
            outcomes.setdefault(label, []).extend(np.atleast_1d(vals))

    expected = {lbl: order.count(lbl) for lbl in dict.fromkeys(order)}
    got = {lbl: len(v) for lbl, v in outcomes.items()}
    if got != expected:
        raise ValueError(
            f"Outcome counts per qubit {got} don't match the circuit's "
            f"measurements {expected}"
        )
    seen = dict.fromkeys(outcomes, 0)
    record = []
    for label in order:
        record.append(outcomes[label][seen[label]])
        seen[label] += 1
    return np.array(record, dtype=bool)


def stim_record_from_outcomes(
    circuit: STIMPhysicalCircuit | stim.Circuit,
    outcomes: ProgramResults | Sequence[Any],
) -> StimRecord:
    """Rebuild STIM's measurement record, detectors and observables.

    Parameters
    ----------
    circuit:
        The circuit that was run. A `stim.Circuit` is wrapped in a
        [](api:STIMPhysicalCircuit) so its qubit labels match the run's.

    outcomes:
        A [](api:ProgramResults), or a sequence of shots. Each shot is a
        mapping from qubit label to outcomes, or a sequence of such mappings
        (or `None`) with one per frame; the outcomes are joined in frame
        order.

    Returns
    -------
    StimRecord
        Boolean arrays of shape `(shots, n)`.

    Raises
    ------
    ValueError
        If a qubit's number of outcomes differs from the circuit's, or a
        qubit the circuit never measures has outcomes.

    NotImplementedError
        If the circuit writes record bits that LoQS can't attribute to a
        qubit, such as `MPP` or `MPAD`.
    """
    if not isinstance(circuit, STIMPhysicalCircuit):
        circuit = STIMPhysicalCircuit(circuit, suppress_tick_warning=True)
    if isinstance(outcomes, ProgramResults):
        all_shots: Any = outcomes.collect_shot_data(
            "measurement_outcomes", "all"
        )
    else:
        all_shots = outcomes

    text = str(circuit.circuit)
    order = _label_order(text, tuple(circuit.qubit_labels))
    shots = [_shot_record(shot, order) for shot in all_shots]
    measurements = np.array(shots, dtype=bool).reshape(len(shots), len(order))
    detectors, observables = _m2d_converter(text).convert(
        measurements=measurements, separate_observables=True
    )
    return StimRecord(measurements, detectors, observables)


@functools.lru_cache(maxsize=32)
def _matching(dem_str: str) -> Any:
    import pymatching

    return pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel(dem_str)
    )


def _stim_decoder_apply_fn(
    circuit_str: str,
    qubit_labels: list,
    dem_str: str,
    measurement_outcomes: Any,
) -> Frame:
    """Decode one shot's outcomes with the DEM's PyMatching decoder."""
    order = _label_order(circuit_str, tuple(qubit_labels))
    record = _shot_record(measurement_outcomes, order)
    detectors, observables = _m2d_converter(circuit_str).convert(
        measurements=record.reshape(1, -1), separate_observables=True
    )
    predicted = _matching(dem_str).decode(detectors[0])
    observed = observables[0]
    return Frame(
        {
            "detectors": [int(d) for d in detectors[0]],
            "observables": [int(o) for o in observed],
            "predicted_observables": [int(o) for o in predicted],
            "logical_error": bool(np.any(predicted != observed)),
        }
    )


def _stim_decoder_batch(
    measurements: NDArray,
    record_labels: Sequence,
    circuit_str: str,
    qubit_labels: Sequence,
    dem_str: str,
) -> dict | None:
    """Decode many shots' records at once, as `_stim_decoder_apply_fn` does.

    `measurements` is `(shots, n)` and `record_labels` the qubit label
    behind each column. Returns `None` when the labels aren't the
    circuit's record order, so the caller falls back to decoding per shot.
    """
    order = _label_order(circuit_str, tuple(qubit_labels))
    if tuple(record_labels) != order:
        return None
    detectors, observables = _m2d_converter(circuit_str).convert(
        measurements=measurements, separate_observables=True
    )
    predicted = _matching(dem_str).decode_batch(detectors)
    observed = observables.astype(np.uint8)
    return {
        "detectors": detectors.astype(np.uint8),
        "observables": observed,
        "predicted_observables": predicted.astype(np.uint8),
        "logical_error": np.any(predicted != observed, axis=1),
    }


def build_stim_decoder_instruction(
    circuit: STIMPhysicalCircuit | stim.Circuit,
    dem: stim.DetectorErrorModel | None = None,
    name: str = "STIM DEM decoder",
) -> Instruction:
    """Build an instruction that decodes a STIM circuit's run with PyMatching.

    It rebuilds the measurement record from the program's measurement
    outcomes, and adds the frame keys `detectors`, `observables`,
    `predicted_observables` (lists of 0/1) and `logical_error` (whether any
    prediction differs from the observed flip). It applies no correction.
    Its data is plain text and lists, so the instruction survives saving and
    loading a program.

    Parameters
    ----------
    circuit:
        The circuit the program runs. A `stim.Circuit` is wrapped in a
        [](api:STIMPhysicalCircuit) so its qubit labels match the run's.

    dem:
        The detector error model to decode with. Defaults to
        `circuit.detector_error_model(decompose_errors=True)`, which is
        empty when the noise comes from a LoQS model rather than the circuit;
        pass the DEM of a noisy copy of the circuit then.

    name:
        Instruction name.

    Raises
    ------
    ValueError
        If the DEM has no error mechanisms, or its detector count differs
        from the circuit's.
    """
    try:
        import pymatching  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyMatching is not installed, cannot use pymatching decoder. "
            "Please install pymatching: pip install loqs[pymatching]"
        ) from e

    if not isinstance(circuit, STIMPhysicalCircuit):
        circuit = STIMPhysicalCircuit(circuit, suppress_tick_warning=True)
    text = str(circuit.circuit)
    if dem is None:
        dem = circuit.circuit.detector_error_model(decompose_errors=True)
    if not any(inst.type == "error" for inst in dem.flattened()):
        raise ValueError(
            "The detector error model has no error mechanisms, so the "
            "decoder would never correct anything. Pass `dem=` from a "
            "noisy copy of the circuit."
        )
    if dem.num_detectors != circuit.circuit.num_detectors:
        raise ValueError(
            f"The DEM has {dem.num_detectors} detectors but the circuit "
            f"has {circuit.circuit.num_detectors}"
        )
    dem_str = str(dem)
    _matching(dem_str)  # fail early

    return Instruction(
        _stim_decoder_apply_fn,
        data={
            "circuit_str": text,
            "qubit_labels": list(circuit.qubit_labels),
            "dem_str": dem_str,
        },
        param_priorities={"measurement_outcomes": ["history[all]"]},
        name=name,
        type="STIM DEM decoder",
    )
