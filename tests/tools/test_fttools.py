"""Tester for loqs.tools.fttools"""

import numpy as np
import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from loqs.backends import PyGSTiPhysicalCircuit
from loqs.core import Frame, QuantumProgram
from loqs.core.instructions import builders
from loqs.core.instructions.instruction import Instruction
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.core.historydatacollector import HistoryDataCollector
from loqs.tools import fttools
from loqs.tools.multiprogramrunner import CheckpointConfig
from loqs.tools.noisesweeptools import NoiseSweepRunner
from loqs.tools.paralleltools import ParallelStrategy


def _build_circuit_program():
    """A minimal, real (but not necessarily run-able) QuantumProgram
    whose single stack entry holds a PyGSTi-backed physical circuit
    instruction: H on Q0, then CNOT(Q0, Q1). Exercises both entries of
    PAULI_PROPAGATION_GATE_MAP.
    """
    circ = PyGSTiPhysicalCircuit([("Gh", "Q0")], qubit_labels=["Q0", "Q1"])
    circ = circ.append([("Gcnot", "Q0", "Q1")])
    inst = builders.build_physical_circuit_instruction(
        circuit=circ, name="Circuit"
    )
    program = QuantumProgram(
        instruction_stack=[{"instruction": "Circuit"}],
        global_instructions={"Circuit": inst},
        name="ft base program",
    )
    return program, inst, circ


def _build_counter_program():
    """A minimal, genuinely run-able QuantumProgram (no physical circuit
    at all) for exercising the run/collect-and-compare machinery."""
    trivial_code = trivial_codepack.create_qec_code()
    qubits = ["Q0"]
    ideal_model = trivial_codepack.create_ideal_model(qubits)
    stack = [
        {
            "instruction": "Init Patch Trivial",
            "new_patch_label": "L0",
            "qubits": qubits,
        },
        {
            "instruction": "Init Counter",
            "patch_label": "L0",
            "initial_value": 0,
        },
        {"instruction": "Increment", "patch_label": "L0", "increment_by": 1},
    ]
    return QuantumProgram(
        stack,
        default_noise_model=ideal_model,
        patch_types={"Trivial": trivial_code},
        name="ft counter test",
    )


def _make_runner(programs, expected_outcomes=(1,), **kwargs):
    kwargs.setdefault("collect_shot_data_args", [("counter", -1)])
    kwargs.setdefault("num_shots", 1)
    return fttools.FaultInjectionRunner(
        errored_programs=programs,
        expected_outcomes=expected_outcomes,
        **kwargs,
    )


class TestBuildDiscreteErrorInjectionProgramForCombo:

    def test_injects_error_and_preserves_rest_of_stack(self):
        program, _, _ = _build_circuit_program()
        new_program = fttools.build_discrete_error_injection_program_for_combo(
            program, 0, [(0, "Gxpi", 0)]
        )
        assert new_program is not program
        assert len(new_program.instruction_stack) == 1
        new_label = new_program.instruction_stack[0]
        assert new_label["instruction"] == "Circuit"
        assert new_label["error_injections"] == [(0, "Gxpi", 0)]
        # The original program/label must be untouched (deepcopy, not alias)
        assert "error_injections" not in program.instruction_stack[0]


class TestPauliPropagation:

    def test_propagate_x_through_hadamard_becomes_z(self):
        _, _, circ = _build_circuit_program()
        # H maps X->Z; a lone Z on the CNOT control commutes through unchanged.
        signature = fttools.propagate_pauli_signature(circ, 0, {0: "X"})
        assert signature == ((0, "Z"),)

    def test_propagate_x_through_cnot_control_spreads_to_target(self):
        _, _, circ = _build_circuit_program()
        # Starting after the H (layer 1, just the CNOT): X on the control
        # propagates to X on both control and target.
        signature = fttools.propagate_pauli_signature(circ, 1, {0: "X"})
        assert signature == ((0, "X"), (1, "X"))

    def test_propagate_unsupported_gate_raises(self):
        circ = PyGSTiPhysicalCircuit([("Gzpi", "Q0")], qubit_labels=["Q0"])
        with pytest.raises(ValueError):
            fttools.propagate_pauli_signature(circ, 0, {0: "X"})

    def test_prune_error_combos_weight_1(self):
        _, _, circ = _build_circuit_program()
        representatives, total = fttools.prune_error_combos_by_propagation(
            circ, ["Gxpi", "Gzpi"], post_twoq_gates=False
        )
        # 3 locations x 2 labels = 6 combos; they propagate to 4 distinct
        # final signatures.
        assert total == 6
        assert len(representatives) == 4

    def test_prune_error_combos_weight_2(self):
        _, _, circ = _build_circuit_program()
        representatives, total = fttools.prune_error_combos_by_propagation(
            circ, ["Gxpi", "Gzpi"], post_twoq_gates=True
        )
        # 1 two-qubit-gate location x 2x2 label combos = 4
        assert total == 4
        assert len(representatives) == 4


class TestBuildPrunedDiscreteErrorInjectionPrograms:

    def test_returns_fewer_or_equal_programs_than_total(self):
        program, inst, _ = _build_circuit_program()
        programs, total = (
            fttools.build_pruned_discrete_error_injection_programs(
                program, inst, 0, ["Gxpi", "Gzpi"], post_twoq_gates=False
            )
        )
        assert total == 6
        assert 0 < len(programs) <= total
        for p in programs:
            assert "error_injections" in p.instruction_stack[0]


class TestBuildDiscreteErrorInjectionPrograms:

    def test_missing_circuit_key_raises(self):
        program, _, _ = _build_circuit_program()

        def apply_fn():
            pass

        bad_inst = Instruction(apply_fn, data={}, name="bad")

        with pytest.raises(ValueError):
            fttools.build_discrete_error_injection_programs(
                program, bad_inst, 0, ["Gxpi"]
            )

    @pytest.mark.parametrize(
        "post_twoq_gates, n, weight",
        [(False, 6, 1), (True, 4, 2)],
        ids=["weight_1", "weight_2"],
    )
    def test_sweep_covers_every_location_and_label(
        self, post_twoq_gates, n, weight
    ):
        """Weight 1 is 3 locations x 2 labels; weight 2 is 1 two-qubit gate
        location x 2x2 label combos."""
        program, inst, _ = _build_circuit_program()
        programs = fttools.build_discrete_error_injection_programs(
            program,
            inst,
            0,
            ["Gxpi", "Gzpi"],
            post_twoq_gates=post_twoq_gates,
        )
        injections = [
            p.instruction_stack[0]["error_injections"] for p in programs
        ]
        assert len(programs) == n
        assert len({tuple(i) for i in injections}) == n
        assert all(len(i) == weight for i in injections)


class TestRunDiscreteErrorInjectedPrograms:

    @pytest.mark.parametrize(
        "expected_outcomes, num_failed",
        [([1], 0), ([999], 2)],
        ids=["all_succeed", "some_fail"],
    )
    def test_failed_programs_are_collected(
        self, expected_outcomes, num_failed
    ):
        program = _build_counter_program()
        runner = _make_runner([program, program], expected_outcomes)
        failed = runner.run()
        assert failed == [program] * num_failed


class TestRunDiscreteErrorInjectedProgramsParallel:
    """[](api:FaultInjectionRunner)'s `parallel_strategy` (a
    [](api:ParallelStrategy)) path, against real `loky` and `submitit`
    executors -- both must return the driver's own original program
    objects in the failed list (per the runner's own contract), not
    copies that crossed a process boundary. `ParallelStrategy`'s own
    construction-time validation (mutual exclusion, `n_program_chunks`
    requirements) is covered directly in test_paralleltools.py, not
    duplicated here."""

    def test_loky_program_executor_failures_are_the_driver_s_own_objects(
        self,
    ):
        loky = pytest.importorskip("loky")
        program = _build_counter_program()
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        runner = _make_runner(
            [program, program], [999], parallel_strategy=strategy
        )
        failed = runner.run()

        assert failed == [program, program]
        assert all(p is program for p in failed)


class TestFaultInjectionRunnerCheckpointing:
    """Checkpoint/resume and crash-recovery tests for FaultInjectionRunner."""

    def test_existing_checkpoint_auto_resumes(self, tmp_path):
        """A resumed call whose expected_outcomes is equivalent but
        differently typed (`(1,)` vs `[1]`) continues from the checkpoint."""
        program = _build_counter_program()
        ckpt = tmp_path / "checkpoint"

        runner1 = _make_runner(
            [program],
            [1],
            config=CheckpointConfig(item_checkpoint_dir=ckpt),
        )
        assert runner1.run() == []

        runner2 = _make_runner(
            [program],
            (1,),
            config=CheckpointConfig(item_checkpoint_dir=ckpt, resume=True),
        )
        assert runner2.run() == []


class TestRunKwargsPassthrough:
    """Test run_kwargs passthrough in FaultInjectionRunner."""

    def test_run_kwargs_roundtrips_via_serialization(
        self, tmp_path, make_temp_path
    ):
        """FaultInjectionRunner with run_kwargs serializes and deserializes correctly."""
        program = _build_counter_program()
        item_ckpt = tmp_path / "item_checkpoint"

        original_runner = _make_runner(
            [program],
            config=CheckpointConfig(item_checkpoint_dir=item_ckpt),
            run_kwargs={"max_frame_limit": 999},
        )

        with make_temp_path(suffix=".h5") as f_path:
            original_runner.write(f_path)
            loaded_runner = fttools.FaultInjectionRunner.read(f_path)

        assert loaded_runner.run_kwargs == {"max_frame_limit": 999}


def _flip_coin_apply(seed, fail_prob=0.0) -> Frame:
    """ "Fail" a shot with probability `fail_prob`, deterministically from `seed`."""
    rng = np.random.default_rng(seed)
    return Frame({"failed": bool(rng.random() < fail_prob)})


FLIP_COIN = Instruction(apply_fn=_flip_coin_apply, name="Flip Coin")


def _fault_injection_runner(collector, config):
    return _make_runner(
        [_build_counter_program()],
        collect_shot_data_args=[collector],
        config=config,
    )


def _noise_sweep_runner(collector, config):
    return NoiseSweepRunner(
        strengths=[0.0, 0.1],
        num_shots=1,
        collect_shot_data_args=[collector],
        expected_outcomes=[False, False],
        instruction_stack=[{"instruction": "Flip Coin", "fail_prob": 0.1}],
        global_instructions={"Flip Coin": FLIP_COIN},
        config=config,
    )


class TestHistoryDataCollectorWithDict:
    """Tests that a literal HistoryDataCollector instance can be used directly
    in collect_shot_data_args and survives checkpointing/serialization."""

    @pytest.mark.parametrize(
        "make_runner, key",
        [
            (_fault_injection_runner, "counter"),
            (_noise_sweep_runner, "failed"),
        ],
        ids=["fault_injection_runner", "noise_sweep_runner"],
    )
    def test_literal_history_data_collector_in_runner_serialize(
        self, tmp_path, make_temp_path, make_runner, key
    ):
        # A literal HistoryDataCollector instance, not a raw dict/tuple spec.
        collector = HistoryDataCollector(key=key, indices=-1)
        runner = make_runner(
            collector,
            CheckpointConfig(item_checkpoint_dir=tmp_path / "item_checkpoint"),
        )
        runner.run()

        # Serialize and resume
        with make_temp_path(suffix=".h5") as f_path:
            runner.write(f_path)
            loaded = type(runner).read(f_path)

        # Loaded instance should have the same collector
        assert loaded.collect_shot_data_args == [collector]


class TestFaultInjectionRunnerHooks:
    """Direct unit tests of FaultInjectionRunner's own derived build_program/
    reduce_program_outcomes/CHKPT_SUBDIR_PREFIX hook implementations, in
    isolation from MultiProgramRunner's generic dispatch/checkpoint/resume/
    parallel machinery -- that generic behavior is covered once, generically,
    in test_multiprogramrunner.py."""

    def test_build_program_returns_item_at_index(self):
        """build_program returns the errored_programs entry at the given
        index directly, with no transformation."""
        program_a = _build_counter_program()
        program_b = _build_counter_program()
        runner = _make_runner([program_a, program_b])
        assert runner.build_program(0) is program_a
        assert runner.build_program(1) is program_b

    @pytest.mark.parametrize(
        "expected_outcome, expected_reduction",
        [(1, True), (999, False)],
        ids=["true_on_match", "false_on_mismatch"],
    )
    def test_reduce_program_outcomes_pass_fail(
        self, expected_outcome, expected_reduction
    ):
        """reduce_program_outcomes returns True when every shot matches the
        expected outcome, and False as soon as any shot mismatches."""
        program = _build_counter_program()
        runner = _make_runner([program], [expected_outcome])
        program_results = program.run(num_shots=1, verbose=False)
        assert (
            runner.reduce_program_outcomes(program_results)
            is expected_reduction
        )

    def test_reduce_program_outcomes_raises_on_missing_shot(self):
        """A results object with fewer shots than `num_shots` raises
        instead of judging only the shots present."""
        program = _build_counter_program()
        runner = _make_runner([program], num_shots=2)
        program_results = program.run(num_shots=1, verbose=False)
        with pytest.raises(RuntimeError):
            runner.reduce_program_outcomes(program_results)

    def test_build_output_treats_none_result_as_failed(self):
        """A None result entry (an incomplete run) is treated as failed by
        ordinary Python truthiness in _build_output."""
        program_ok = _build_counter_program()
        program_incomplete = _build_counter_program()
        runner = _make_runner([program_ok, program_incomplete])
        ordered_results = [(program_ok, True), (program_incomplete, None)]

        assert runner._build_output(ordered_results) == [program_incomplete]
