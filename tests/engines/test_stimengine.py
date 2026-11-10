"""Tester for loqs.engines.StimEngine."""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from loqs.backends import (
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMPhysicalCircuit,
    STIMQuantumState,
    propagate_state,
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
    HistoryDataCollector,
    InstructionStack,
    MeasurementOutcomes,
    OutputResults,
    QECCode,
    QuantumProgram,
)
from loqs.core.instructions import InstructionLabel
from loqs.core.instructions.builders import (
    build_composite_instruction,
    build_lookup_decoder_instruction,
    build_object_builder_instruction,
    build_physical_circuit_instruction,
)
from loqs.core.instructions.instruction import Instruction
from loqs.engines import StimEngine
from loqs.tools.stimtools import (
    _label_order,
    _shot_record,
    build_stim_decoder_instruction,
    build_stim_model,
    stim_record_from_outcomes,
)

NOISE = dict(
    after_clifford_depolarization=0.005,
    after_reset_flip_probability=0.005,
    before_measure_flip_probability=0.005,
)


# TODO: Clean this up after #100 lands
def split_program(
    rounds=3, state_type=STIMQuantumState, seed=None, decoder=False
):
    """A noisy surface-code program split at TICKs into several steps.

    With `decoder`, a DEM decoder step follows the circuits.
    """
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
    if decoder:
        insts.append(
            build_stim_decoder_instruction(
                full, dem=noisy.detector_error_model(decompose_errors=True)
            )
        )
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
        program, _, _ = split_program(rounds=2, decoder=True)
        compiled = StimEngine().compile(program)
        assert compiled.steps[-1]["reads"]
        with make_temp_path(suffix=f".{format}") as f_path:
            compiled.write(f_path)
            loaded = CompiledProgram.read(f_path)
        assert loaded.qubit_labels == compiled.qubit_labels
        assert loaded.steps == compiled.steps


class TestRefusals:

    @staticmethod
    def _with_step(apply_fn):
        """`simple_program`'s circuits, then one classical step."""
        program = simple_program("X 0\nM 0\n", [0])
        extra = InstructionLabel(Instruction(apply_fn, name="classical"))
        stack = list(program.instruction_stack) + [extra]
        return QuantumProgram.from_quantum_program(
            program, InstructionStack(stack)
        )

    @classmethod
    def _stack_reader(cls):
        return cls._with_step(lambda stack: Frame({}))

    @classmethod
    def _stack_writer(cls):
        return cls._with_step(lambda: Frame({"stack": InstructionStack([])}))

    @classmethod
    def _state_reader(cls):
        return cls._with_step(
            lambda state: Frame({"n": len(state.qubit_labels)})
        )

    @classmethod
    def _state_reader_in_composite(cls):
        step = Instruction(
            lambda state: Frame({"n": len(state.qubit_labels)}),
            name="classical",
        )
        composite = build_composite_instruction([step], name="wrapped")
        program = simple_program("X 0\nM 0\n", [0])
        stack = list(program.instruction_stack) + [InstructionLabel(composite)]
        return QuantumProgram.from_quantum_program(
            program, InstructionStack(stack)
        )

    @classmethod
    def _composite_forwards_outcomes(cls):
        step = Instruction(
            lambda measurement_outcomes: Frame(
                {"bit": int(measurement_outcomes[0][0])}
            ),
            name="classical",
        )
        composite = build_composite_instruction([step], name="forwarding")
        program = simple_program("X 0\nM 0\n", [0])
        stack = list(program.instruction_stack) + [InstructionLabel(composite)]
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
    def _custom_circuit():
        program = simple_program("X 0\n", [0, 1])
        stack = list(program.instruction_stack) + [
            InstructionLabel(_peeking_instruction())
        ]
        return QuantumProgram.from_quantum_program(
            program, InstructionStack(stack)
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
            "_probabilistic",
            "_unreported",
            "_state_replaced",
            "_plain_state_builder",
            "_flipped_history",
            "_stack_reader",
            "_stack_writer",
            "_state_reader",
            "_state_reader_in_composite",
            "_composite_forwards_outcomes",
            "_custom_circuit",
        ],
        ids=[
            "statevector-state",
            "probabilistic-rep",
            "unreported-bits",
            "state-replaced",
            "plain-state-builder",
            "flipped-initial-history",
            "stack-reader",
            "stack-writer",
            "state-reader",
            "state-reader-in-composite",
            "composite-forwards-outcomes",
            "custom-circuit",
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
        import pymatching

        n = 100_000
        program, noisy, _ = split_program(seed=1, decoder=True)
        decoder = program.instruction_stack[-1]["instruction"]
        calls = []
        apply_fn = decoder.apply_fn

        def counted(*args, **kwargs):
            calls.append(1)
            return apply_fn(*args, **kwargs)

        decoder.apply_fn = counted
        results = StimEngine().run(
            program, n, outputs=["logical_error", "predicted_observables"]
        )
        det, obs = noisy.compile_detector_sampler(seed=2).sample(
            n, separate_observables=True
        )
        matching = pymatching.Matching.from_detector_error_model(
            noisy.detector_error_model(decompose_errors=True)
        )
        expected = np.any(matching.decode_batch(det) != obs, axis=1)

        assert results["measurements"].shape == (n, noisy.num_measurements)
        assert results["detectors"].shape == (n, noisy.num_detectors)
        assert results["observables"].shape == (n, noisy.num_observables)
        assert results["logical_error"].shape == (n,)
        assert results["predicted_observables"].shape == (n, 1)
        assert_agree(results["detectors"], det)
        assert_agree(results["observables"], obs)
        assert_agree(
            results["logical_error"].reshape(-1, 1), expected.reshape(-1, 1)
        )
        assert len(calls) < n

    @pytest.mark.slow
    def test_batched_against_per_shot(self):
        program, _, full = split_program(seed=3, decoder=True)
        per_shot = program.run(3_000, verbose=False)
        record = stim_record_from_outcomes(full, per_shot)
        results = StimEngine().run(program, 100_000, outputs=["logical_error"])
        errors = np.asarray(
            per_shot.collect_shot_data("logical_error", -1), dtype=float
        )

        assert_agree(results["detectors"], record.detectors)
        assert_agree(results["observables"], record.observables)
        assert_agree(
            results["logical_error"].reshape(-1, 1), errors.reshape(-1, 1)
        )

    def test_batched_decoder_in_replay(self, monkeypatch):
        memory, _, _ = split_program(seed=1, decoder=True)
        decoder = memory.instruction_stack[-1]["instruction"]
        decodes, replays = [], []
        apply_fn, replay = decoder.apply_fn, StimEngine._replay

        def counted(*args, **kwargs):
            decodes.append(1)
            return apply_fn(*args, **kwargs)

        def spy(*args, **kwargs):
            replays.append(1)
            return replay(*args, **kwargs)

        def copy_detectors(detectors):
            return Frame({"seen": detectors})

        decoder.apply_fn = counted
        monkeypatch.setattr(StimEngine, "_replay", spy)
        extra = InstructionLabel(Instruction(copy_detectors, name="copy"))
        stack = list(memory.instruction_stack) + [extra]
        program = QuantumProgram.from_quantum_program(
            memory, InstructionStack(stack)
        )
        n = 10
        results = StimEngine().run(program, n, outputs=["seen"])
        assert np.array_equal(results["seen"], results["detectors"])
        assert len(decodes) < n

        replays.clear()
        collector = {
            "key": "logical_error",
            "indices": "all",
            "strip_none_entries": True,
        }
        StimEngine().run(memory, n, outputs=[collector])
        assert len(replays) < n

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

    @pytest.mark.parametrize("path", ["batched", "fast-path"])
    def test_same_seed_same_arrays(self, path):
        if path == "batched":
            program, _, _ = split_program(seed=7)
            outputs, keys = None, ("measurements", "detectors", "observables")
            shots = 1_000
        else:
            program = _outcome_dependent_program(seed=7)
            outputs, keys = _FAST_OUTPUTS, ("measurements", "branch")
            shots = 10
        first = StimEngine().run(program, shots, outputs=outputs)
        second = StimEngine().run(program, shots, outputs=outputs)
        for key in keys:
            for a, b in zip(first.records(key), second.records(key)):
                assert np.array_equal(a, b)

    def test_read_back_program_runs(self, tmp_path):
        path = str(tmp_path / "program.json")
        simple_program("X 0\nM 0\n", [0]).write(path)
        records = StimEngine().run(QuantumProgram.read(path), 3)
        assert np.array_equal(records["measurements"], [[1]] * 3)


def _assert_record(expected, got):
    """A shot's record equals `collect_shot_data`'s, at every level."""
    if isinstance(got, list):
        assert len(got) == len(expected)
        for e, g in zip(expected, got):
            _assert_record(e, g)
    else:
        assert np.array_equal(np.asarray(expected), got)


def _replayed(program, shots, collectors):
    """Per-shot results, and the engine's outputs on the same shots.

    Each shot's record is rebuilt from its frames' outcomes in the
    compiled steps' order, then fed to the shot replay.
    """
    engine = StimEngine()
    plan = engine._plan(program)
    per_shot = program.run(shots, verbose=False)
    order = tuple(
        lbl for s in plan.compiled.steps for lbl in s["outcome_labels"]
    )
    rows = np.array(
        [
            _shot_record(
                [f.get("measurement_outcomes") for f in history], order
            )
            for history in (per_shot.shot_histories[i] for i in range(shots))
        ]
    )
    resolved = engine._resolve(plan, collectors)
    out = engine._outputs(plan, rows, resolved)
    results = OutputResults(out)
    for col in collectors:
        key = col["key"]
        expected = HistoryDataCollector.from_raw(col).collect(per_shot)
        records = results.records(key)
        assert len(records) == shots
        for e, g in zip(expected, records):
            _assert_record(e, g)
    return plan, results


def _track(patch_label, patches, seed):
    data = patches[patch_label].data
    data["rounds"] = data.get("rounds", 0) + 1
    return Frame({"rounds": data["rounds"], "draw": seed})


def _readout(patch_label, patches, measurement_outcomes):
    frame = patches[patch_label].pauli_frame
    bits = [
        int(measurement_outcomes[q][0]) ^ frame.get_bit("X", q)
        for q in ("D0", "D1", "D2")
    ]
    return Frame({"corrected": bits, "logical": int(sum(bits) >= 2)})


def _feed_forward_program():
    labels = ["D0", "D1", "D2", "A0", "A1"]

    def circuit(text, name, **kwargs):
        physical = STIMPhysicalCircuit(
            text, qubit_labels=labels, suppress_tick_warning=True
        )
        return InstructionLabel(
            build_physical_circuit_instruction(physical, name=name, **kwargs),
            patch_label="L0",
        )

    table = {"00": "III", "10": "XII", "11": "IXI", "01": "IIX"}
    decode = InstructionLabel(
        build_lookup_decoder_instruction(
            table,
            [("A0", -2, 0), ("A1", -2, 0)],
            "raw_syn",
            diff_prev_syndrome=True,
            name="Decode",
        ),
        patch_label="L0",
    )
    track = InstructionLabel(
        Instruction(_track, name="Track"), patch_label="L0"
    )
    readout = InstructionLabel(
        Instruction(
            _readout,
            param_priorities={"measurement_outcomes": ["history[-1]"]},
            name="Readout",
        ),
        patch_label="L0",
    )
    syndrome = lambda: circuit(  # noqa: E731
        "CX 0 3 1 3 1 4 2 4\nTICK\nMR 3 4\n", "SE"
    )
    code = QECCode({}, labels, ["D0", "D1", "D2"])
    stack = [
        {"instruction": "Init State", "state": 5, "qubit_labels": labels},
        {
            "instruction": "Init Patch Rep",
            "new_patch_label": "L0",
            "qubits": labels,
        },
        syndrome(),
        track,
        decode,
        circuit("I 0 1 2\n", "Flip", pauli_frame_update="H"),
        syndrome(),
        track,
        decode,
        circuit("M 0 1 2\n", "Measure data"),
        readout,
    ]
    model = build_stim_model(
        stim.Circuit("CX 0 1\nI 0\nMR 0\nM 0\n"),
        after_clifford_depolarization=0.1,
    )
    return QuantumProgram(
        stack,
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={"Rep": code},
        default_base_seed=11,
    )


class TestFeedForward:

    def test_lookup_decoder_shot_for_shot(self):
        program = _feed_forward_program()
        keys = [
            "raw_syn",
            "syndrome_diff",
            "rounds",
            "draw",
            "corrected",
            "logical",
        ]
        collectors = [
            {"key": k, "indices": "all", "strip_none_entries": True}
            for k in keys
        ]
        plan, results = _replayed(program, 50, collectors)

        assert results["raw_syn"].shape == (50, 2, 2)
        # The readout depends on both decoders
        decoders = [
            i
            for i, s in enumerate(plan.compiled.steps)
            if s["name"] == "Decode"
        ]
        assert len(decoders) == 2
        assert set(decoders) <= set(plan.compiled.steps[-1]["reads"])

    def test_steane_codepack_shot_for_shot(self):
        pytest.importorskip("pygsti")
        from loqs.backends import ListPhysicalCircuit
        from loqs.backends.reps import StimCircuitGateRep
        from loqs.codepacks import codepack_7_1_3_quantinuum2021 as steane
        from loqs.core.instructions import builders

        qubits = ["A0", "A1", "A2"] + [f"D{i+3}" for i in range(7)]
        code = steane.create_qec_code(circuit_backend=ListPhysicalCircuit)
        model = steane.create_ideal_model(
            qubits, gaterep=StimCircuitGateRep, model_backend=DictNoiseModel
        )
        error = ListPhysicalCircuit([[("Gxpi", "D3")]], qubit_labels=qubits)
        code.instructions["Injected Data Error"] = (
            builders.build_physical_circuit_instruction(
                error, name="Injected Data Error"
            )
        )

        def rnd():
            return [
                ("Unflagged Parallel S1-S2-S3 Check", "L0"),
                ("Unflagged Parallel S4-S5-S6 Check", "L0"),
                ("QEC Decoder", "L0"),
            ]

        stack = [
            {
                "instruction": "Init State",
                "state": len(qubits),
                "qubit_labels": qubits,
            },
            {
                "instruction": "Init Patch Steane",
                "new_patch_label": "L0",
                "qubits": qubits,
            },
            ("Non-FT Zero Prep", "L0"),
            ("Injected Data Error", "L0"),
            *rnd(),
            ("H", "L0"),
            *rnd(),
            ("H", "L0"),
            ("X", "L0"),
            *rnd(),
            ("FT Logical Z Measure", "L0"),
        ]
        program = QuantumProgram(
            InstructionStack(stack),
            default_noise_model=model,
            state_type=STIMQuantumState,
            patch_types={"Steane": code},
            default_base_seed=5,
        )
        plan = StimEngine()._plan(program)
        # Every key a classical step sets whose dry-run values are numeric
        values: dict = {}
        for step in plan.steps:
            if step.kind == "classical":
                i = plan.steps.index(step) - plan.n_setup + 1
                for k in step.keys:
                    values.setdefault(k, []).append(plan.dry[i].get(k))
        keys = []
        for k, vals in values.items():
            arrs = [StimEngine._numeric(v) for v in vals]
            if all(a is not None for a in arrs) and (
                len({a.ndim for a in arrs}) == 1
            ):
                keys.append(k)
        assert "logical_measurement" in keys
        collectors = [
            {"key": k, "indices": "all", "strip_none_entries": True}
            for k in keys
        ]
        _replayed(program, 3, collectors)


class TestOutputRefusals:

    @pytest.mark.parametrize(
        "outputs, error",
        [
            (
                [
                    {
                        "key": "logical_error",
                        "indices": "all",
                        "frame_filter": {"logical_error": True},
                    }
                ],
                NotImplementedError,
            ),
            (["detectors"], ValueError),
            ([{"key": "state", "indices": 0}], NotImplementedError),
        ],
        ids=["per-shot-filter", "engine-name", "setup-frame"],
    )
    def test_refused_before_any_shot(self, outputs, error):
        pytest.importorskip("pymatching")
        program, _, _ = split_program(decoder=True)
        with pytest.raises(error):
            StimEngine().run(program, 2, outputs=outputs)


def _branch(measurement_outcomes, stack):
    """Insert circuit `D` at the front of the stack when qubit 0 read 1."""
    bit = int(measurement_outcomes[0][0])
    if bit:
        stack = stack.insert_instruction(0, InstructionLabel(_branch.circuit))
    return Frame({"stack": stack, "branch": bit, "copy": bit})


def _outcome_dependent_stack():
    """Circuits, then a step that inserts a circuit when a bit reads 1."""
    labels = [0, 1, 2]

    def circuit(text, name):
        return build_physical_circuit_instruction(
            STIMPhysicalCircuit(
                text, qubit_labels=labels, suppress_tick_warning=True
            ),
            name=name,
        )

    plain = circuit("MR 1\n", "C")
    _branch.circuit = circuit("X 2\nTICK\nM 2\n", "D")
    return [
        {"instruction": "Init State", "state": 3, "qubit_labels": labels},
        InstructionLabel(plain, error_injections=[(0, "X", 1)]),
        InstructionLabel(build_composite_instruction([plain], name="plain C")),
        InstructionLabel(circuit("H 0\nTICK\nMR 0\n", "H")),
        InstructionLabel(Instruction(_branch, name="Branch")),
    ]


def _outcome_dependent_program(*tail, seed=None):
    model = build_stim_model(stim.Circuit("H 0\nX 0\nM 0\nMR 0\n"))
    return QuantumProgram(
        _outcome_dependent_stack() + [InstructionLabel(t) for t in tail],
        default_noise_model=model,
        state_type=STIMQuantumState,
        default_base_seed=seed,
    )


def _state_user(state):
    return Frame({"n": len(state.qubit_labels)})


def _rebuild():
    return Frame({"state": STIMQuantumState(1)})


def _rebuilding_reader(seed, stack):
    if seed % 2:
        stack = stack.insert_instruction(
            0, InstructionLabel(Instruction(_rebuild, name="rebuild"))
        )
    return Frame({"stack": stack})


def _peeking_circuit(state, model):
    """A physical circuit whose own code reads the state."""
    text = "X 1\nM 1\n" if state.state.peek_z(0) == -1 else "M 1\n"
    circuit = STIMPhysicalCircuit(
        text, qubit_labels=[0, 1], suppress_tick_warning=True
    )
    new_state, outcomes = propagate_state(circuit, model, state, True)
    return Frame(
        {
            "state": new_state,
            "measurement_outcomes": MeasurementOutcomes(outcomes),
            "applied_stim_circuit_str": str(new_state.latest_applied_circuit),
        }
    )


def _peeking_instruction():
    return Instruction(
        _peeking_circuit, name="peeking", type="Physical circuit"
    )


def _peeking_reader(seed, stack):
    if seed % 2:
        stack = stack.insert_instruction(
            0, InstructionLabel(_peeking_instruction())
        )
    return Frame({"stack": stack})


def _looping_reader(seed, stack):
    if seed % 2:
        stack = stack.insert_instruction(
            0, InstructionLabel(Instruction(_looping_reader, name="loop"))
        )
    return Frame({"stack": stack})


_FAST_OUTPUTS = [
    {"key": "branch", "indices": "all", "strip_none_entries": True},
    {"key": "copy", "indices": 5},
]


class TestFastPath:

    @staticmethod
    def _choice_program(operations):
        """`X 1`, `M 1 0` on labels `[1, 0]`, with `X` drawn from `operations`."""
        rep = ProbabilisticStimGateRep(operations, (0,))
        model = DictNoiseModel(
            {"X": rep},
            {"M": StimCircuitInstrumentRep("M 0", ())},
            gatereps=[ProbabilisticStimGateRep],
            instreps=[StimCircuitInstrumentRep],
        )
        program = simple_program("X 1\nTICK\nM 1 0\n", [1, 0])
        return QuantumProgram.from_quantum_program(
            program, default_noise_model=model, default_base_seed=3
        )

    def test_choice_node(self):
        program = self._choice_program([("X 0", 0.2), ("I 0", 0.8)])
        n = 2000
        records = StimEngine().run(program, n)["measurements"]
        bits = np.asarray(records, dtype=int)
        # Label order is [1, 0]; the X on label 1 is the drawn gate
        mean = bits[:, 0].mean()
        sigma = np.sqrt(0.2 * 0.8 / n)
        assert abs(mean - 0.2) <= 3 * sigma
        assert not bits[:, 1].any()

    def test_measuring_alternative_refused(self):
        # However rare, a measuring alternative would shift the record
        program = self._choice_program([("X 0\nM 0", 1e-6), ("I 0", 1 - 1e-6)])
        with pytest.raises(NotImplementedError):
            StimEngine().run(program, 4)

    def test_label_parameters_key_the_cache(self):
        labels = [0, 1]
        mr = build_physical_circuit_instruction(
            STIMPhysicalCircuit(
                "MR 1\n", qubit_labels=labels, suppress_tick_warning=True
            ),
            name="C",
        )
        program = QuantumProgram(
            [
                {
                    "instruction": "Init State",
                    "state": 2,
                    "qubit_labels": labels,
                },
                InstructionLabel(mr, error_injections=[(0, "X", 1)]),
                InstructionLabel(mr),
                InstructionLabel(
                    Instruction(
                        lambda stack: Frame({"stack": stack}), name="r"
                    )
                ),
            ],
            default_noise_model=build_stim_model(stim.Circuit("X 0\nMR 0\n")),
            state_type=STIMQuantumState,
            default_base_seed=1,
        )
        records = StimEngine().run(program, 3)["measurements"]
        assert np.array_equal(records, [[1, 0]] * 3)

    def test_outcome_dependent_order(self, monkeypatch):
        calls = []
        compile_circuit = StimEngine._fast_circuit

        def spy(self, *args, **kwargs):
            calls.append(1)
            return compile_circuit(self, *args, **kwargs)

        monkeypatch.setattr(StimEngine, "_fast_circuit", spy)
        results = StimEngine().run(
            _outcome_dependent_program(seed=1), 10, outputs=_FAST_OUTPUTS
        )
        assert len(calls) <= 4
        for rec, branch, copy in zip(
            results.records("measurements"),
            results.records("branch"),
            results.records("copy"),
        ):
            bit = int(rec[2])
            assert list(rec) == [1, 0, bit] + [1] * bit
            assert list(branch) == [bit]
            assert copy == bit

    @pytest.mark.parametrize(
        "apply_fn, engine, error",
        [
            (_state_user, StimEngine(), NotImplementedError),
            (_rebuilding_reader, StimEngine(), NotImplementedError),
            (_looping_reader, StimEngine(max_frame_limit=10), RuntimeError),
            (_peeking_reader, StimEngine(), NotImplementedError),
        ],
        ids=["state-reader", "state-rebuilt", "frame-limit", "custom-circuit"],
    )
    def test_refused(self, apply_fn, engine, error):
        program = _outcome_dependent_program(
            Instruction(apply_fn, name="tail"), seed=0
        )
        with pytest.raises(error):
            engine.run(program, 2)

    @pytest.mark.parametrize(
        "pack, damped, gate",
        [
            ("steane", None, None),
            ("five-qubit", "Gh", None),
            ("five-qubit", None, "K"),
        ],
        ids=[
            "steane-ft",
            "five-qubit-flagged-damped",
            "five-qubit-frame-update",
        ],
    )
    def test_codepack_shot_for_shot(self, pack, damped, gate):
        program, qubits, keys = _codepack_program(pack, 11, damped, gate=gate)
        shots = 10
        collectors = _codepack_collectors(keys)
        per_shot = program.run(shots, verbose=False)
        results = StimEngine().run(program, shots, outputs=collectors)

        for col in collectors:
            expected = HistoryDataCollector.from_raw(col).collect(per_shot)
            records = results.records(col["key"])
            assert len(records) == shots
            for e, g in zip(expected, records):
                _assert_record(e, g)
        for i, rec in enumerate(results.records("measurements")):
            history = per_shot.shot_histories[i]
            assert np.array_equal(rec, _frame_records(history, qubits))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "pack, syndrome",
        [("steane", "unflagged_syndrome"), ("five-qubit", "raw_syndrome")],
        ids=["steane-ft", "five-qubit-flagged"],
    )
    def test_codepack_agrees_with_per_shot(self, pack, syndrome):
        reference, _, _ = _codepack_program(pack, 1)
        program, _, _ = _codepack_program(pack, 100_000)
        collectors = [
            {"key": "logical_measurement", "indices": -1},
            *[
                {"key": k, "indices": "all", "strip_none_entries": True}
                for k in ("total_rus_count", syndrome)
            ],
        ]
        per_shot = reference.run(1000, verbose=False)
        results = StimEngine().run(program, 5000, outputs=collectors)

        def quantities(records, key):
            if key == "logical_measurement":
                return [float(np.asarray(r)) for r in records]
            if key == "total_rus_count":
                return [float(np.sum(r)) for r in records]
            return [float(len(r)) for r in records]

        for col in collectors:
            key = col["key"]
            expected = HistoryDataCollector.from_raw(col).collect(per_shot)
            x = np.array(quantities(expected, key))[:, None]
            y = np.array(quantities(results.records(key), key))[:, None]
            assert_agree(x, y)


_STEANE_QUBITS = ["A0", "A1", "A2"] + [f"D{i+3}" for i in range(7)]
_FIVE_QUBITS = ["A0", "A1"] + [f"D{i+2}" for i in range(5)]
_STEANE_KEYS = [
    "classical_syndrome",
    "flag_syndrome_diff",
    "unflagged_syndrome",
    "new_logical_pauli_frame",
]
_FIVE_KEYS = ["raw_syndrome", "syndrome_diff", "raw_M1", "inferred_M1"]


def _noisy_model(ideal, p, damped_gate=None, damping=0.02):
    """The ideal model with depolarizing and readout noise at rate `p`.

    Each gate's text is followed by a depolarizing channel on its targets
    and each measurement is `X_ERROR(p)` then `MR`. A `damped_gate` is a
    `ProbabilisticStimGateRep` of its noisy text and a reset.
    """
    gates = {}
    for (name, qubits), rep in ideal.gate_dict.items():
        n = len(qubits)
        channel = "DEPOLARIZE1" if n == 1 else "DEPOLARIZE2"
        text = (
            f"{rep.circuit_str}\n{channel}({p}) {' '.join(map(str, range(n)))}"
        )
        if name == damped_gate:
            gates[(name, qubits)] = ProbabilisticStimGateRep(
                [(text, 1 - damping), ("R 0", damping)], qubits
            )
        else:
            gates[(name, qubits)] = StimCircuitGateRep(text, qubits)
    insts = {
        key: StimCircuitInstrumentRep(f"X_ERROR({p}) 0\nMR 0", key[1])
        for key in ideal.inst_dict
    }
    return DictNoiseModel(
        gates,
        insts,
        gatereps=[StimCircuitGateRep, ProbabilisticStimGateRep],
        instreps=[StimCircuitInstrumentRep],
    )


def _codepack_program(pack, seed, damped_gate=None, damping=0.02, gate=None):
    """A noisy FT program of a codepack: (program, qubits, collector keys).

    A `gate` (a logical gate that updates the Pauli frame) runs just before
    the logical measurement.
    """
    pytest.importorskip("pygsti")
    from loqs.backends import ListPhysicalCircuit

    if pack == "steane":
        from loqs.codepacks import codepack_7_1_3_quantinuum2021 as mod

        qubits, name, keys = _STEANE_QUBITS, "Steane", _STEANE_KEYS
        body = [
            ("FT Zero Prep", "L0"),
            ("Adaptive QEC", "L0"),
            ("Adaptive QEC", "L0"),
            ("FT Logical Z Measure", "L0"),
        ]
    else:
        from loqs.codepacks import codepack_5_1_3_quantinuum2022 as mod

        qubits, name, keys = _FIVE_QUBITS, "5Q", _FIVE_KEYS
        body = [
            ("FT Minus Prep", "L0"),
            ("Flagged QEC", "L0"),
            ("FT Logical X Measure", "L0"),
        ]
    if gate is not None:
        body.insert(-1, (gate, "L0"))
    code = mod.create_qec_code(circuit_backend=ListPhysicalCircuit)
    ideal = mod.create_ideal_model(
        qubits, gaterep=StimCircuitGateRep, model_backend=DictNoiseModel
    )
    model = _noisy_model(ideal, 0.01, damped_gate, damping)
    stack = [
        {
            "instruction": "Init State",
            "state": len(qubits),
            "qubit_labels": qubits,
        },
        {
            "instruction": "Init Patch " + name,
            "new_patch_label": "L0",
            "qubits": qubits,
        },
        *body,
    ]
    program = QuantumProgram(
        InstructionStack(stack),
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={name: code},
        default_base_seed=seed,
    )
    return program, qubits, keys


def _codepack_collectors(keys):
    return [{"key": "logical_measurement", "indices": -1}] + [
        {"key": k, "indices": "all", "strip_none_entries": True}
        for k in ["total_rus_count", *keys]
    ]


def _frame_records(history, qubits):
    """A per-shot history's measurement record, frame by frame."""
    bits = []
    for frame in history:
        text = frame.get("applied_stim_circuit_str")
        if text is None:
            continue
        order = _label_order(text, tuple(qubits))
        outcomes = frame.get("measurement_outcomes") or {}
        bits.extend(_shot_record(outcomes, order))
    return np.array(bits, dtype=bool)
