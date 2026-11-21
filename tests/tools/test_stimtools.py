"""Tester for loqs.tools.stimtools."""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("quantumsim")

from loqs.backends import (
    NumpyStatevectorQuantumState,
    QSimQuantumState,
    STIMPhysicalCircuit,
    STIMQuantumState,
)
from loqs.backends.reps import (
    KrausGateRep,
    PTMGateRep,
    QSimSuperopGateRep,
    UnitaryGateRep,
    convert as convert_rep,
)
from loqs.backends.reps.base import RepConstructionError
from loqs.core import QuantumProgram
from loqs.core.instructions import InstructionLabel
from loqs.core.instructions.builders import build_physical_circuit_instruction
from loqs.tools.stimtools import (
    _label_order,
    _stim_decoder_apply_fn,
    _stim_decoder_batch,
    build_stim_decoder_instruction,
    build_stim_model,
    stim_record_from_outcomes,
)

STATES = {
    "stim": STIMQuantumState,
    "statevector": NumpyStatevectorQuantumState,
    "qsim": QSimQuantumState,
}

GATEREPS = {
    "stim": {},
    "statevector": {"gaterep": KrausGateRep},
    "qsim": {"gaterep": QSimSuperopGateRep},
}

# Deterministic on a noiseless run; the X on qubit 5 is injected at layer 2.
# Qubit 6 reaches +Y through a gate, so the Y basis's sign is checked too,
# and the CX checks the two-qubit gates' qubit order.
ALL_KINDS = """R 0 5 6
RX 1
RY 2
R 3 4
TICK
X 3
H 4
SQRT_X_DAG 6
TICK
CX 3 0
TICK
M 0 5
MX 1
MY 2 6
MR 3
MRX 4
DETECTOR rec[-7]
DETECTOR rec[-6]
OBSERVABLE_INCLUDE(0) rec[-6]
"""


def run(circuit, model, state_type, shots, injections=None):
    physical = STIMPhysicalCircuit(circuit, suppress_tick_warning=True)
    inst = build_physical_circuit_instruction(physical, name="circuit")
    label = InstructionLabel(inst, error_injections=injections)
    program = QuantumProgram(
        [
            {
                "instruction": "Init State",
                "state": len(physical.qubit_labels),
                "qubit_labels": physical.qubit_labels,
            },
            label,
        ],
        default_noise_model=model,
        state_type=state_type,
    )
    return physical, program.run(num_shots=shots, verbose=False)


class TestBuildStimModel:

    @pytest.mark.parametrize(
        "task",
        [
            "repetition_code:memory",
            "surface_code:rotated_memory_x",
            "color_code:memory_xyz",
        ],
    )
    def test_noise_matches_generated(self, task):
        noise = dict(
            after_clifford_depolarization=0.01,
            after_reset_flip_probability=0.02,
            before_measure_flip_probability=0.03,
        )
        ideal = stim.Circuit.generated(task, distance=3, rounds=2)
        noisy = stim.Circuit.generated(task, distance=3, rounds=2, **noise)
        model = build_stim_model(ideal, **noise)
        _, results = run(ideal, model, STIMQuantumState, 1)
        applied = results.collect_shot_data(
            "applied_stim_circuit_str", "all", strip_none_entries=True
        )[0][0]
        dem = stim.Circuit(applied).detector_error_model()
        assert dem.approx_equals(noisy.detector_error_model(), atol=1e-12)

    @pytest.mark.parametrize("state", STATES)
    def test_record_of_every_instrument(self, state):
        circuit = stim.Circuit(ALL_KINDS)
        model = build_stim_model(
            circuit,
            **GATEREPS[state],
        )
        physical, results = run(
            circuit, model, STATES[state], 3, injections=[(2, "X", 5)]
        )
        record = stim_record_from_outcomes(circuit, results)
        expected = circuit.reference_sample().copy()
        expected[1] ^= True
        assert np.array_equal(record.measurements, [expected] * 3)
        det, obs = circuit.compile_m2d_converter().convert(
            measurements=record.measurements, separate_observables=True
        )
        assert np.array_equal(record.detectors, det)
        assert np.array_equal(record.observables, obs)
        assert record.detectors.any() and record.observables.any()

    def test_dense_kraus_weights(self):
        circuit = stim.Circuit("H 0\nCX 0 1\n")
        p = 0.1
        model = build_stim_model(
            circuit,
            gaterep=KrausGateRep,
            after_clifford_depolarization=p,
        )
        for name, n in (("H", 1), ("CX", 2)):
            rep = model.gate_dict[name]
            ptm = convert_rep(rep, PTMGateRep).ptm
            unitary = stim.Tableau.from_named_gate(name).to_unitary_matrix(
                endian="big"
            )
            ideal = convert_rep(
                UnitaryGateRep(unitary, rep.qubit_labels), PTMGateRep
            )
            depol = np.diag([1.0] + [1 - p * 4**n / (4**n - 1)] * (4**n - 1))
            assert np.allclose(ptm, depol @ ideal.ptm)

    @pytest.mark.parametrize(
        "noise",
        ["after_reset_flip_probability", "before_measure_flip_probability"],
    )
    def test_dense_flips_match_across_states(self, noise):
        circuit = stim.Circuit(ALL_KINDS)
        records = {}
        for state in STATES:
            model = build_stim_model(circuit, **{noise: 1}, **GATEREPS[state])
            _, results = run(circuit, model, STATES[state], 1)
            records[state] = stim_record_from_outcomes(circuit, results)
        noiseless = circuit.reference_sample()
        for record in records.values():
            assert np.array_equal(
                record.measurements, records["stim"].measurements
            )
            assert not np.array_equal(record.measurements[0], noiseless)

    def test_noisy_unitary_rep_refused(self):
        with pytest.raises(RepConstructionError):
            build_stim_model(
                stim.Circuit("H 0\n"),
                gaterep=UnitaryGateRep,
                after_clifford_depolarization=0.1,
            )

    def test_bad_inputs_refused(self):
        with pytest.raises(ValueError):
            build_stim_model(
                stim.Circuit("H 0\n"), after_clifford_depolarization=2
            )
        with pytest.raises(NotImplementedError, match="MXX"):
            build_stim_model(stim.Circuit("MXX 0 1\n"))
        with pytest.raises(NotImplementedError, match="MPP"):
            build_stim_model(
                stim.Circuit("MPP Z0*Z1\n"),
                before_measure_flip_probability=0.1,
            )


class TestRecordFromOutcomes:

    def test_mismatched_outcomes_refused(self):
        circuit = stim.Circuit("M 0 1\nDETECTOR rec[-1]\n")
        with pytest.raises(ValueError):
            stim_record_from_outcomes(circuit, [{0: [0], 1: [0, 1]}])
        with pytest.raises(ValueError):
            stim_record_from_outcomes(circuit, [{0: [0], 1: [0], 2: [1]}])

    @pytest.mark.parametrize("text", ["MPAD 0\nM 0\n", "MPP Z0\nM 0\n"])
    def test_unattributable_record_bits_refused(self, text):
        circuit = stim.Circuit(text)
        with pytest.raises(NotImplementedError):
            stim_record_from_outcomes(circuit, [{0: [0]}])

    def test_frames_are_joined_in_order(self):
        circuit = stim.Circuit("M 0 1\nM 0\n")
        record = stim_record_from_outcomes(
            circuit, [[{0: [1], 1: [0]}, None, {0: [0]}]]
        )
        assert record.measurements.tolist() == [[True, False, False]]


class TestDecoderInstruction:

    @pytest.mark.parametrize(
        "format, injections, want",
        [
            (None, [(1, "X", 4)], ([1], [1], False)),
            ("json", [(1, "X", 4)], ([1], [1], False)),
            ("hdf5", [(1, "X", 4)], ([1], [1], False)),
            (None, [(1, "X", 0), (1, "X", 2)], ([0], [1], True)),
        ],
        ids=["corrected", "json", "hdf5", "logical-error"],
    )
    def test_decodes_injected_error(
        self, format, injections, want, make_temp_path
    ):
        noise = dict(
            after_clifford_depolarization=0.01,
            after_reset_flip_probability=0.01,
            before_measure_flip_probability=0.01,
        )
        task = "repetition_code:memory"
        ideal = stim.Circuit.generated(task, distance=3, rounds=2)
        noisy = stim.Circuit.generated(task, distance=3, rounds=2, **noise)
        physical = STIMPhysicalCircuit(ideal, suppress_tick_warning=True)
        program = QuantumProgram(
            [
                {
                    "instruction": "Init State",
                    "state": len(physical.qubit_labels),
                    "qubit_labels": physical.qubit_labels,
                },
                InstructionLabel(
                    build_physical_circuit_instruction(physical, name="c"),
                    error_injections=injections,
                ),
                build_stim_decoder_instruction(
                    ideal,
                    dem=noisy.detector_error_model(decompose_errors=True),
                ),
            ],
            default_noise_model=build_stim_model(ideal),
            state_type=STIMQuantumState,
        )
        if format is not None:
            with make_temp_path(suffix=f".{format}") as path:
                program.write(path)
                program = QuantumProgram.read(path)
        results = program.run(num_shots=3, verbose=False)
        keys = ("observables", "predicted_observables", "logical_error")
        for key, value in zip(keys, want):
            got = results.collect_shot_data(
                key, "all", strip_none_entries=True
            )
            assert all(g[-1] == value for g in got)
        dets = results.collect_shot_data(
            "detectors", "all", strip_none_entries=True
        )
        assert all(any(d[-1]) for d in dets)

    def test_batch_matches_per_shot(self):
        noise = dict(
            after_clifford_depolarization=0.005,
            after_reset_flip_probability=0.005,
            before_measure_flip_probability=0.005,
        )
        task = "surface_code:rotated_memory_z"
        ideal = stim.Circuit.generated(task, distance=3, rounds=3)
        noisy = stim.Circuit.generated(task, distance=3, rounds=3, **noise)
        inst = build_stim_decoder_instruction(
            ideal, dem=noisy.detector_error_model(decompose_errors=True)
        )
        data = inst.data
        order = _label_order(data["circuit_str"], tuple(data["qubit_labels"]))
        shots = noisy.compile_sampler(seed=5).sample(200)

        batch = _stim_decoder_batch(shots, order, **data)
        assert batch is not None
        for i, row in enumerate(shots):
            outcomes: dict = {}
            for label, bit in zip(order, row):
                outcomes.setdefault(label, []).append(int(bit))
            frame = _stim_decoder_apply_fn(
                measurement_outcomes=outcomes, **data
            )
            assert set(frame) == set(batch)
            for key in frame:
                assert np.array_equal(frame[key], batch[key][i])

        assert _stim_decoder_batch(shots, order[::-1], **data) is None

    def test_noiseless_circuit_without_dem_refused(self):
        ideal = stim.Circuit.generated(
            "repetition_code:memory", distance=3, rounds=2
        )
        with pytest.raises(ValueError):
            build_stim_decoder_instruction(ideal)
