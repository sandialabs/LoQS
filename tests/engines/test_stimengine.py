"""Tester for loqs.engines.StimEngine."""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from loqs.backends import (
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMPhysicalCircuit,
    STIMQuantumState,
)
from loqs.backends.reps import (
    ProbabilisticStimGateRep,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.core import (
    CompiledProgram,
    Frame,
    History,
    InstructionStack,
    QuantumProgram,
)
from loqs.core.instructions import InstructionLabel
from loqs.core.instructions.builders import (
    build_object_builder_instruction,
    build_physical_circuit_instruction,
)
from loqs.core.instructions.instruction import Instruction
from loqs.engines import StimEngine
from loqs.tools.stimtools import build_stim_model, stim_record_from_outcomes

NOISE = dict(
    after_clifford_depolarization=0.005,
    after_reset_flip_probability=0.005,
    before_measure_flip_probability=0.005,
)

# TODO: Clean this up after #100 lands
def split_program(rounds=3, state_type=STIMQuantumState, seed=None):
    """A noisy surface-code program split at TICKs into several steps."""
    ideal = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=rounds
    )
    noisy = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=rounds, **NOISE
    )
    full = STIMPhysicalCircuit(ideal.flattened(), suppress_tick_warning=True)
    labels = full.qubit_labels
    chunks, cur, ticks = [], [], 0
    for line in str(full.circuit).split("\n"):
        cur.append(line)
        if line.strip() == "TICK":
            ticks += 1
            if ticks % 4 == 0:
                chunks.append("\n".join(cur))
                cur = []
    if cur:
        chunks.append("\n".join(cur))
    assert len(chunks) >= 4
    insts = [
        build_physical_circuit_instruction(
            STIMPhysicalCircuit(
                stim.Circuit(c),
                qubit_labels=labels,
                suppress_tick_warning=True,
            ),
            name=f"chunk {i}",
        )
        for i, c in enumerate(chunks)
    ]
    program = QuantumProgram(
        [
            {
                "instruction": "Init State",
                "state": len(labels),
                "qubit_labels": labels,
            }
        ]
        + [InstructionLabel(i) for i in insts],
        default_noise_model=build_stim_model(ideal, **NOISE),
        state_type=state_type,
        default_base_seed=seed,
    )
    return program, noisy, full


def joined(compiled):
    return stim.Circuit(
        "\n".join(s["emitted"]["stim"] for s in compiled.steps if s["emitted"])
    )


def sorted_dem(circuit):
    return sorted(str(i) for i in circuit.detector_error_model().flattened())


def simple_program(circuit_text, labels, state_labels=None, **kwargs):
    """A program of one circuit, with a zero-error model."""
    physical = STIMPhysicalCircuit(
        circuit_text, qubit_labels=labels, suppress_tick_warning=True
    )
    inst = build_physical_circuit_instruction(physical, name="circuit")
    stack = [
        {
            "instruction": "Init State",
            "state": len(labels),
            "qubit_labels": state_labels or labels,
        },
        InstructionLabel(inst),
    ]
    model = build_stim_model(stim.Circuit("X 0\nM 0\nR 0\n"))
    return QuantumProgram(
        stack,
        default_noise_model=model,
        state_type=STIMQuantumState,
        **kwargs,
    )


class TestCompile:

    def test_trace_is_exact(self):
        program, noisy, _ = split_program()
        compiled = StimEngine().compile(program)

        assert isinstance(compiled, CompiledProgram)
        text = joined(compiled)
        assert text.num_measurements == noisy.num_measurements
        assert text.num_detectors == noisy.num_detectors
        assert sorted_dem(text) == sorted_dem(noisy)
        assert (
            sum(len(s["outcome_labels"]) for s in compiled.steps)
            == noisy.num_measurements
        )

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_compiled_program_round_trip(self, format, make_temp_path):
        program, _, _ = split_program(rounds=2)
        compiled = StimEngine().compile(program)
        with make_temp_path(suffix=f".{format}") as f_path:
            compiled.write(f_path)
            loaded = CompiledProgram.read(f_path)
        assert loaded.qubit_labels == compiled.qubit_labels
        assert loaded.steps == compiled.steps


class TestRefusals:

    @staticmethod
    def _classical():
        program, _, _ = split_program()
        classical = Instruction(lambda: Frame({}), name="classical")
        stack = list(program.instruction_stack) + [InstructionLabel(classical)]
        return QuantumProgram.from_quantum_program(
            program, InstructionStack(stack)
        )

    @staticmethod
    def _probabilistic():
        rep = ProbabilisticStimGateRep([("X 0", 0.5), ("I 0", 0.5)], (0,))
        model = DictNoiseModel(
            {"X": rep},
            {"M": StimCircuitInstrumentRep("M 0", ())},
            gatereps=[ProbabilisticStimGateRep],
            instreps=[StimCircuitInstrumentRep],
        )
        program = simple_program("X 0\nTICK\nM 0\n", [0])
        return QuantumProgram.from_quantum_program(
            program, default_noise_model=model
        )

    @staticmethod
    def _unreported():
        model = DictNoiseModel(
            {"X": StimCircuitGateRep("X 0", ())},
            {
                "R": ZBasisProjectionInstrumentRep(0, False, (0,)),
                "M": StimCircuitInstrumentRep("M 0", ()),
            },
            gatereps=[StimCircuitGateRep],
            instreps=[ZBasisProjectionInstrumentRep, StimCircuitInstrumentRep],
        )
        program = simple_program("R 0\nTICK\nX 0\nTICK\nM 0\n", [0])
        return QuantumProgram.from_quantum_program(
            program, default_noise_model=model
        )

    @staticmethod
    def _state_replaced():
        program = simple_program("X 0\nM 0\n", [0])
        stack = list(program.instruction_stack)
        init, circuit = stack
        return QuantumProgram.from_quantum_program(
            program, InstructionStack([init, circuit, init, circuit])
        )

    @staticmethod
    def _statevector():
        program, _, _ = split_program(state_type=NumpyStatevectorQuantumState)
        return program

    @staticmethod
    def _plain_state_builder():
        program = simple_program("X 0\nM 0\n", [0])
        builder = build_object_builder_instruction("state", STIMQuantumState)
        stack = [
            InstructionLabel(builder, state=1),
            list(program.instruction_stack)[1],
        ]
        return QuantumProgram(
            stack, default_noise_model=program.default_noise_model
        )

    @staticmethod
    def _flipped_history():
        program = simple_program("X 0\nM 0\n", [0, 1])
        circuit = list(program.instruction_stack)[1]
        history = History(
            [Frame({"state": STIMQuantumState([0, 1]), "stack": [circuit]})]
        )
        return QuantumProgram(
            [circuit],
            initial_history=history,
            default_noise_model=program.default_noise_model,
        )

    @pytest.mark.parametrize(
        "make",
        [
            "_statevector",
            "_classical",
            "_probabilistic",
            "_unreported",
            "_state_replaced",
            "_plain_state_builder",
            "_flipped_history",
        ],
        ids=[
            "statevector-state",
            "classical-instruction",
            "probabilistic-rep",
            "unreported-bits",
            "state-replaced",
            "plain-state-builder",
            "flipped-initial-history",
        ],
    )
    def test_refused(self, make):
        program = getattr(self, make)()
        with pytest.raises(NotImplementedError):
            StimEngine().compile(program)

    def test_bit_list_init_state_refused(self):
        program = simple_program("X 0\nM 0\n", [0, 1])
        circuit = list(program.instruction_stack)[1]
        init = {"instruction": "Init State", "state": [1, 0]}
        program = QuantumProgram.from_quantum_program(
            program, InstructionStack([init, circuit])
        )
        with pytest.raises(ValueError) as exc:
            StimEngine().compile(program)
        assert isinstance(exc.value.__cause__, NotImplementedError)

    def test_frame_limit_truncation_raises(self):
        program, _, _ = split_program()
        with pytest.warns(UserWarning), pytest.raises(RuntimeError):
            StimEngine(max_frame_limit=2).compile(program)


def assert_agree(a, b, sigmas=3):
    """Per-shot sums of two `(shots, n)` samples agree within `sigmas`."""
    x, y = a.sum(axis=1).astype(float), b.sum(axis=1).astype(float)
    sigma = np.sqrt(x.var() / len(x) + y.var() / len(y))
    assert abs(x.mean() - y.mean()) <= sigmas * sigma


class TestRun:

    def test_batched_against_stim(self):
        n = 100_000
        program, noisy, _ = split_program(seed=1)
        results = StimEngine().run(program, n)
        det, obs = noisy.compile_detector_sampler(seed=2).sample(
            n, separate_observables=True
        )

        assert results["measurements"].shape == (n, noisy.num_measurements)
        assert results["detectors"].shape == (n, noisy.num_detectors)
        assert results["observables"].shape == (n, noisy.num_observables)
        assert_agree(results["detectors"], det)
        assert_agree(results["observables"], obs)

    @pytest.mark.slow
    def test_batched_against_per_shot(self):
        program, _, full = split_program(seed=3)
        per_shot = program.run(3_000, verbose=False)
        record = stim_record_from_outcomes(full, per_shot)
        results = StimEngine().run(program, 100_000)

        assert_agree(results["detectors"], record.detectors)
        assert_agree(results["observables"], record.observables)

    @pytest.mark.parametrize("kind", ["init-state", "initial-history"])
    def test_permuted_integer_labels(self, kind):
        labels = [7, 3, 11, 0, 5]
        state_labels = [5, 0, 11, 3, 7]
        parts = [
            STIMPhysicalCircuit(
                text, qubit_labels=labels, suppress_tick_warning=True
            )
            for text in ("X 0 2\nTICK\nM 0 1\n", "X 4\nTICK\nM 2 3 4\n")
        ]
        insts = [
            InstructionLabel(
                build_physical_circuit_instruction(p, name=f"part {i}")
            )
            for i, p in enumerate(parts)
        ]
        model = build_stim_model(stim.Circuit("X 0\nM 0\n"))
        if kind == "init-state":
            program = QuantumProgram(
                [
                    {
                        "instruction": "Init State",
                        "state": 5,
                        "qubit_labels": state_labels,
                    }
                ]
                + insts,
                default_noise_model=model,
                state_type=STIMQuantumState,
            )
        else:
            state = STIMQuantumState(5, qubit_labels=state_labels)
            program = QuantumProgram(
                insts,
                initial_history=History(
                    [Frame({"state": state, "stack": list(insts)})]
                ),
                default_noise_model=model,
            )

        compiled = StimEngine().compile(program)
        batched = StimEngine().run(program, 3)["measurements"]
        per_shot = program.run(3, verbose=False)
        order = [lbl for s in compiled.steps for lbl in s["outcome_labels"]]
        for shot in range(3):
            expected: dict = {}
            for frame in per_shot.shot_histories[shot]:
                for lbl, vals in (
                    frame.get("measurement_outcomes") or {}
                ).items():
                    expected.setdefault(lbl, []).extend(int(v) for v in vals)
            got: dict = {}
            for lbl, bit in zip(order, batched[shot]):
                got.setdefault(lbl, []).append(int(bit))
            assert got == expected

    def test_same_seed_same_arrays(self):
        program, _, _ = split_program(seed=7)
        first = StimEngine().run(program, 1_000)
        second = StimEngine().run(program, 1_000)
        for key in ("measurements", "detectors", "observables"):
            assert np.array_equal(first[key], second[key])
