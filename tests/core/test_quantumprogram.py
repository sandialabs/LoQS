"""Tester for loqs.core.quantumprogram"""

import pytest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# TODO: Rework to run with native backend only if failure to import
quantumsim = pytest.importorskip("quantumsim")
stim = pytest.importorskip("stim")

from loqs.backends import QSimQuantumState, STIMQuantumState
from loqs.core import Instruction, QuantumProgram
from loqs.core.frame import Frame
from loqs.core.programresults import ProgramResults
from loqs.internal.checkpointstore import (
    SHOT_LAYOUT,
    CheckpointReadError,
    CheckpointStore,
)
from loqs.internal.serializable import Serializable
from loqs.core.history import History
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from tests.tools._shared_checkpoint_test_helpers import (
    InlineExecutor,
    inline_execution_patches,
)


def _init_counter_from_seed(initial_value):
    """Apply function seeding a counter from the shot's own seed, so a
    shot's result can be tied back to its originating shot index."""
    return Frame({"counter": initial_value})


# A global instruction whose "initial_value" is sourced from the shot's
# own seed, so each shot's result depends on its own index.
_SEED_COUNTER_INSTRUCTION = Instruction(
    _init_counter_from_seed,
    param_priorities={"initial_value": ["program"]},
    param_aliases={"initial_value": "seed"},
    name="Initialize counter from shot seed",
)


def _build_seeded_counter_program(state_type=None, name="seeded counter"):
    """A program whose shot `i` ends with `counter == i + 1`. With a
    `state_type`, the program also initializes that state."""
    trivial_code = trivial_codepack.create_qec_code()
    qubits = ["Q0"]
    stack = [
        {
            "instruction": "Init Patch Trivial",
            "new_patch_label": "L0",
            "qubits": qubits,
        },
        {"instruction": "Init Counter From Seed"},
        {
            "instruction": "Increment",
            "patch_label": "L0",
            "increment_by": 1,
        },
    ]
    kwargs = {}
    if state_type is not None:
        stack.insert(
            0,
            {
                "instruction": "Init State",
                "state": len(qubits),
                "qubit_labels": qubits,
            },
        )
        kwargs["state_type"] = state_type
    return QuantumProgram(
        stack,
        default_noise_model=trivial_codepack.create_ideal_model(qubits),
        patch_types={"Trivial": trivial_code},
        global_instructions={
            "Init Counter From Seed": _SEED_COUNTER_INSTRUCTION
        },
        default_base_seed=0,
        name=name,
        **kwargs,
    )


def _build_state_program(
    name, state_type=QSimQuantumState, increments=None, **kwargs
):
    """A program that initializes a state and a patch. With `increments`
    (a sequence, possibly empty) it also initializes a counter and
    increments it by each value."""
    trivial_code = trivial_codepack.create_qec_code()
    qubits = ["Q0"]
    stack = [
        {
            "instruction": "Init State",
            "state": len(qubits),
            "qubit_labels": qubits,
        },
        {
            "instruction": "Init Patch Trivial",
            "new_patch_label": "L0",
            "qubits": qubits,
        },
    ]
    if increments is not None:
        stack.append(
            {
                "instruction": "Init Counter",
                "patch_label": "L0",
                "initial_value": 0,
            }
        )
        for increment_by in increments:
            stack.append(
                {
                    "instruction": "Increment",
                    "patch_label": "L0",
                    "increment_by": increment_by,
                }
            )
    return QuantumProgram(
        stack,
        default_noise_model=trivial_codepack.create_ideal_model(qubits),
        state_type=state_type,
        patch_types={"Trivial": trivial_code},
        name=name,
        **kwargs,
    )


def _assert_wall_clock_times(results, num_shots):
    assert set(results.shot_wall_clock_times) == set(range(num_shots))
    for value in results.shot_wall_clock_times.values():
        assert isinstance(value, float)
        assert value > 0


def _spy_run_shot(monkeypatch, crash_after=None):
    """Record the shot indices `QuantumProgram._run_shot` computes. With
    `crash_after`, the shot after that many computed ones raises."""
    calls = []
    original = QuantumProgram._run_shot

    def spy(self, max_frame_limit, seed, shot_index):
        calls.append(shot_index)
        if crash_after is not None and len(calls) > crash_after:
            raise RuntimeError("Simulated crash")
        return original(self, max_frame_limit, seed, shot_index)

    monkeypatch.setattr(QuantumProgram, "_run_shot", spy)
    return calls


def _forbid_shot_history_decode(monkeypatch):
    """Make decoding a `History` that holds frames raise. The parent
    program's own (empty) histories may still decode."""
    real_decode = Serializable.decode

    def decode(*args, **kwargs):
        value = real_decode(*args, **kwargs)
        if isinstance(value, History) and len(value) > 0:
            raise AssertionError("a shot History was decoded")
        return value

    monkeypatch.setattr(Serializable, "decode", staticmethod(decode))


@pytest.fixture
def inline_patches():
    with inline_execution_patches():
        yield


class TestQuantumProgram:

    # TODO: This is currently only an integrated test depending on a codepack
    # Make actual unit tests, and probably convert this to a codepack test

    # Note that tests will be labeled state-model-circuit with this stacking
    @pytest.mark.parametrize(
        "state_backend", [QSimQuantumState, STIMQuantumState]
    )
    def test_integrated(self, state_backend):
        program = _build_state_program(
            "Trivial counter test",
            state_type=state_backend,
            increments=(1, 2),
        )

        program_results = program.run()

        assert len(program_results.shot_histories) == 1
        assert len(program_results.shot_histories[0]) > 0

    @pytest.mark.parametrize("format", ["json", "hdf5"])
    def test_program_serialization_parameterized(self, format, make_temp_path):
        program = _build_state_program("Trivial counter test", increments=(1,))
        stack_length = len(program.instruction_stack)
        patch_type_count = len(program.patch_types)

        # Sanity check: program can run
        new_program = QuantumProgram.from_quantum_program(program)
        assert len(new_program.run().shot_histories) == 1

        with make_temp_path(suffix=f".{format}") as tempf_path:
            program.write(tempf_path)
            loaded_program = QuantumProgram.read(tempf_path)
            assert isinstance(loaded_program, QuantumProgram)

            assert len(loaded_program.run().shot_histories) == 1
            assert loaded_program.name == "Trivial counter test"
            assert len(loaded_program.instruction_stack) == stack_length
            assert loaded_program.state_type == QSimQuantumState
            assert loaded_program.patch_types is not None
            assert len(loaded_program.patch_types) == patch_type_count

        with make_temp_path(suffix=f".{format}") as f_path:
            loaded_program.write(f_path)
            loaded_program2 = QuantumProgram.read(f_path)
            assert isinstance(loaded_program2, QuantumProgram)

            assert len(loaded_program2.run().shot_histories) == 1
            assert loaded_program2.name == "Trivial counter test"
            assert len(loaded_program2.instruction_stack) == stack_length

            # Only 1 shot is requested by default, so asking for 2 changes it
            assert len(loaded_program2.run(num_shots=2).shot_histories) == 2


class TestExecutorParallelism:
    """`QuantumProgram.run(shot_executor=...)` dispatch. The loky test is
    the process-boundary path; the batch-count test runs in-process."""

    def test_loky_executor_matches_serial_shot_outcomes(self):
        # Counters are seeded per shot index, so matching the serial run
        # also checks indices; the STIM state must survive the trip back.
        loky = pytest.importorskip("loky")

        num_shots = 4
        serial_results = _build_seeded_counter_program(
            state_type=STIMQuantumState
        ).run(num_shots=num_shots, verbose=False)

        executor = loky.get_reusable_executor(max_workers=2)
        parallel_results = _build_seeded_counter_program(
            state_type=STIMQuantumState
        ).run(num_shots=num_shots, shot_executor=executor, verbose=False)

        serial_counters = {
            i: h[-1]["counter"]
            for i, h in serial_results.shot_histories.items()
        }
        parallel_counters = {
            i: h[-1]["counter"]
            for i, h in parallel_results.shot_histories.items()
        }
        assert parallel_counters == serial_counters
        assert set(parallel_results.shot_histories) == set(range(num_shots))
        for history in parallel_results.shot_histories.values():
            assert isinstance(history[-1]["state"], STIMQuantumState)
        _assert_wall_clock_times(parallel_results, num_shots)

    @pytest.mark.usefixtures("inline_patches")
    def test_n_shot_batches_auto_default_dispatches_correct_batch_count(
        self, tmp_path
    ):
        # num_shots=25 with n_shot_batches=None auto-defaults to
        # ceil(25 / 20) = 2 batches, and run() submits that many.
        executor = InlineExecutor()
        results = _build_seeded_counter_program().run(
            num_shots=25,
            shot_executor=executor,
            n_shot_batches=None,
            checkpoint=True,
            checkpoint_dir=tmp_path / "checkpoint",
            checkpoint_batch_size=None,
            lazy_loading=False,
            verbose=False,
        )

        assert len(results.shot_histories) == 25
        assert executor.submit_count == 2

    @pytest.mark.usefixtures("inline_patches")
    @pytest.mark.parametrize(
        "checkpoint",
        [True, False],
        ids=["checkpointed", "not_checkpointed"],
    )
    def test_thread_executor_refused_only_when_checkpointing(
        self, tmp_path, checkpoint
    ):
        run_kwargs = {"num_shots": 4, "verbose": False}
        if checkpoint:
            run_kwargs["checkpoint"] = True
            run_kwargs["checkpoint_dir"] = tmp_path / "ckpt"
        with ThreadPoolExecutor(max_workers=1) as pool:
            program = _build_seeded_counter_program()
            if checkpoint:
                with pytest.raises(ValueError):
                    program.run(shot_executor=pool, **run_kwargs)
                assert not (tmp_path / "ckpt").exists()
            else:
                results = program.run(shot_executor=pool, **run_kwargs)
                assert len(results.shot_histories) == 4


class TestShotWallClockTimes:
    def test_populates_shot_wall_clock_times(self):
        num_shots = 4
        results = _build_seeded_counter_program().run(
            num_shots=num_shots, verbose=False
        )
        _assert_wall_clock_times(results, num_shots)


class TestProgressCallback:
    """`QuantumProgram.run(progress_callback=...)` reports absolute shot
    counts on every dispatch path. Runs use `verbose=False`, where a
    disabled tqdm bar's own counter stays at 0."""

    @pytest.mark.usefixtures("inline_patches")
    @pytest.mark.parametrize(
        "parallel, checkpointed",
        [(False, False), (True, True)],
        ids=["serial_no_checkpoint", "executor_checkpointed"],
    )
    def test_reports_absolute_counts(self, tmp_path, parallel, checkpointed):
        num_shots = 6
        calls = []
        run_kwargs = {
            "num_shots": num_shots,
            "verbose": False,
            "progress_callback": lambda done, total: calls.append(
                (done, total)
            ),
        }
        if checkpointed:
            run_kwargs["checkpoint"] = True
            run_kwargs["checkpoint_dir"] = tmp_path / "checkpoints"
            run_kwargs["checkpoint_batch_size"] = 2
        if parallel:
            run_kwargs["shot_executor"] = InlineExecutor()

        _build_seeded_counter_program().run(**run_kwargs)

        assert calls
        assert calls[0] == (0, num_shots)
        assert calls[-1] == (num_shots, num_shots)
        done_counts = [done for done, _ in calls]
        assert done_counts == sorted(done_counts)
        assert all(total == num_shots for _, total in calls)
        # More than the initial and final calls: intermediate advances fire.
        assert len(calls) > 2


class TestResolveInstructionLegacyNameHint:
    """`_resolve_instruction` on an unknown global instruction, called
    against a minimal duck-typed `self` (only `global_instructions` is
    read on the global-lookup branch)."""

    def test_global_instruction_not_found_raises(self):
        from types import SimpleNamespace

        from loqs.core.instructions import InstructionLabel

        fake_self = SimpleNamespace(global_instructions={})
        label = InstructionLabel("SomeOtherName")
        with pytest.raises(RuntimeError):
            QuantumProgram._resolve_instruction(fake_self, label, frame={})

    def test_run_with_checkpoint_dir_does_not_create_default_checkpoints(
        self, monkeypatch, tmp_path
    ):
        """Regression test: run(..., checkpoint_dir=...) should not create a
        ./checkpoints directory relative to cwd -- only the explicit
        checkpoint_dir should be used."""
        monkeypatch.chdir(tmp_path)
        program = _build_state_program("Checkpoint dir test", increments=())
        checkpoint_dir = tmp_path / "my_checkpoints"

        program.run(
            num_shots=2,
            checkpoint=True,
            checkpoint_batch_size=1,
            checkpoint_dir=checkpoint_dir,
        )

        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "results.h5").exists()
        assert not Path("./checkpoints").exists()


class TestResumeFromCheckpoint:
    """Test resume-from-checkpoint functionality."""

    @pytest.mark.parametrize(
        "first_batch, first_lazy, resume_batch, resume_lazy",
        [
            (2, False, 2, False),
            (1, False, 2, False),
            (2, True, 2, False),
            (2, True, 2, True),
        ],
        ids=[
            "eager",
            "batch_size_change",
            "eager_resume_of_lazy_crash",
            "lazy_resume",
        ],
    )
    def test_resume_after_crash(
        self,
        monkeypatch,
        tmp_path,
        first_batch,
        first_lazy,
        resume_batch,
        resume_lazy,
    ):
        program = _build_seeded_counter_program()
        checkpoint_dir = tmp_path / "checkpoints"

        # First run: crash after 2 of 4 shots
        _spy_run_shot(monkeypatch, crash_after=2)
        with pytest.raises(RuntimeError):
            program.run(
                num_shots=4,
                checkpoint=True,
                checkpoint_batch_size=first_batch,
                checkpoint_dir=checkpoint_dir,
                lazy_loading=first_lazy,
                verbose=False,
            )
        monkeypatch.undo()
        assert (checkpoint_dir / "results.h5").exists()

        # Resume: only the 2 missing shots are recomputed
        calls = _spy_run_shot(monkeypatch)
        results = program.run(
            num_shots=4,
            checkpoint=True,
            resume=True,
            checkpoint_batch_size=resume_batch,
            checkpoint_dir=checkpoint_dir,
            lazy_loading=resume_lazy,
            verbose=False,
        )
        monkeypatch.undo()

        assert len(calls) == 2
        _assert_wall_clock_times(results, 4)
        if not resume_lazy:
            assert len(results.shot_histories) == 4
            for idx in range(4):
                assert results.shot_histories[idx][-1]["counter"] == idx + 1

    @pytest.mark.parametrize(
        "run_kwargs",
        [{"num_shots": 6}, {"num_shots": 4, "max_frame_limit": 200}],
        ids=["num_shots", "max_frame_limit"],
    )
    def test_resume_config_mismatch_raises(self, tmp_path, run_kwargs):
        program = _build_state_program("Mismatch test", increments=())
        checkpoint_dir = tmp_path / "checkpoints"
        program.run(
            num_shots=4,
            max_frame_limit=100,
            checkpoint=True,
            checkpoint_batch_size=2,
            checkpoint_dir=checkpoint_dir,
            lazy_loading=False,
        )

        resume_kwargs = {"max_frame_limit": 100, **run_kwargs}
        with pytest.raises(ValueError):
            program.run(
                checkpoint=True,
                resume=True,
                checkpoint_batch_size=2,
                checkpoint_dir=checkpoint_dir,
                lazy_loading=False,
                **resume_kwargs,
            )

    def test_no_content_in_checkpoint_dir_raises_file_exists_error_fresh_run(
        self, tmp_path
    ):
        """A fresh run (resume=False) hits the foreign-content check before
        the resume flag is read."""
        program = _build_state_program("Invalid checkpoint test")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "unrelated.txt").write_text("some content")

        with pytest.raises(FileExistsError):
            program.run(
                num_shots=2,
                checkpoint=True,
                resume=False,
                checkpoint_batch_size=1,
                checkpoint_dir=checkpoint_dir,
            )

    @pytest.mark.usefixtures("inline_patches")
    def test_resume_with_executor_and_non_contiguous_gaps(
        self, monkeypatch, tmp_path
    ):
        """Resume with an executor: stage non-contiguous done-shot gaps,
        verify only missing shots are computed, and consolidation occurs."""
        program = _build_seeded_counter_program()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create results.h5 with program config (must match run() defaults)
        ProgramResults(
            name="Config snapshot",
            parent_program=program,
            num_shots=6,
            max_frame_limit=100,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
            lazy_loading=False,
        )

        # Stage a partial checkpoint: shots {0,1} and {4,5} done, {2,3}
        # missing (checkpoint_batch_size=2, num_shots=6)
        for shot_idx in [0, 1, 4, 5]:
            results_worker = ProgramResults(
                name=f"Worker {shot_idx}", lazy_loading=False
            )
            history = History()
            history.append(Frame({"counter": shot_idx + 1}))
            results_worker.add_shot(shot_idx, history, wall_clock_time=0.05)
            results_worker.checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id=f"w{shot_idx}"
            )

        # n_shot_batches=3 derives checkpoint_batch_size=ceil(6/3)=2
        calls = _spy_run_shot(monkeypatch)
        results = program.run(
            num_shots=6,
            checkpoint=True,
            resume=True,
            n_shot_batches=3,
            checkpoint_dir=checkpoint_dir,
            shot_executor=InlineExecutor(),
            lazy_loading=False,
            verbose=False,
        )

        assert sorted(calls) == [2, 3]
        assert len(results.shot_histories) == 6
        for shot_idx in range(6):
            counter = results.shot_histories[shot_idx][-1]["counter"]
            assert counter == shot_idx + 1

        # Worker files were cleaned up and results.h5 consolidated
        leftovers = CheckpointStore(checkpoint_dir, SHOT_LAYOUT)
        assert not leftovers.payload_paths()
        assert not leftovers.ledger_paths()
        assert (checkpoint_dir / "results.h5").exists()
        _assert_wall_clock_times(results, 6)

    def test_resume_scattered_workers_no_checkpoint_consolidates(
        self, tmp_path
    ):
        """Resume when results.h5 exists but no consolidated shots yet, only
        scattered worker files: verify consolidation occurs without recomputing.
        """
        program = _build_seeded_counter_program()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Stage a fresh results.h5 with the config but no shot data
        ProgramResults(
            name="Config snapshot",
            parent_program=program,
            num_shots=4,
            max_frame_limit=100,
            checkpoint_enabled=True,
            checkpoint_dir=checkpoint_dir,
            lazy_loading=False,
        )

        for worker_id, shots in (("w0", [0, 1]), ("w1", [2, 3])):
            results_worker = ProgramResults(lazy_loading=False)
            for shot_idx in shots:
                history = History()
                history.append(Frame({"counter": shot_idx + 1}))
                results_worker.add_shot(shot_idx, history)
            results_worker.checkpoint(
                checkpoint_dir=checkpoint_dir, worker_id=worker_id
            )

        # Pre-condition: results.h5 exists, but only worker files hold shots
        assert (checkpoint_dir / "results.h5").exists()
        worker_files_before = CheckpointStore(
            checkpoint_dir, SHOT_LAYOUT
        ).ledger_paths()
        assert len(worker_files_before) == 2

        results = program.run(
            num_shots=4,
            checkpoint=True,
            resume=True,
            checkpoint_batch_size=2,
            checkpoint_dir=checkpoint_dir,
            lazy_loading=False,
            verbose=False,
        )

        assert len(results.shot_histories) == 4
        for shot_idx in range(4):
            counter = results.shot_histories[shot_idx][-1]["counter"]
            assert counter == shot_idx + 1
        assert (checkpoint_dir / "results.h5").exists()
        leftovers = CheckpointStore(checkpoint_dir, SHOT_LAYOUT)
        assert not leftovers.payload_paths()
        assert not leftovers.ledger_paths()

    @pytest.mark.parametrize(
        "run_kwargs",
        [
            {"checkpoint": False},
            {"checkpoint": True, "checkpoint_batch_size": 2},
        ],
        ids=["resume_without_checkpoint", "nothing_to_resume"],
    )
    def test_resume_misuse_raises(self, tmp_path, run_kwargs):
        program = _build_seeded_counter_program()

        with pytest.raises(ValueError):
            program.run(
                num_shots=4,
                resume=True,
                checkpoint_dir=tmp_path / "nonexistent_dir",
                verbose=False,
                **run_kwargs,
            )

    def test_checkpoint_without_resume_raises_when_content_exists(
        self, tmp_path
    ):
        program = _build_seeded_counter_program()
        checkpoint_dir = tmp_path / "checkpoints"

        program.run(
            num_shots=2,
            checkpoint=True,
            checkpoint_batch_size=2,
            checkpoint_dir=checkpoint_dir,
            lazy_loading=False,
            verbose=False,
        )
        assert (checkpoint_dir / "results.h5").exists()

        with pytest.raises(ValueError):
            program.run(
                num_shots=2,
                checkpoint=True,
                resume=False,
                checkpoint_batch_size=2,
                checkpoint_dir=checkpoint_dir,
                verbose=False,
            )

    def test_lazy_resume_of_complete_run_decodes_no_history(
        self, tmp_path, monkeypatch
    ):
        """Resuming a finished checkpoint lazily decodes no History."""
        program = _build_seeded_counter_program()
        program.run(num_shots=5, checkpoint=True, checkpoint_dir=tmp_path)

        _forbid_shot_history_decode(monkeypatch)
        results = program.run(
            num_shots=5,
            checkpoint=True,
            resume=True,
            checkpoint_dir=tmp_path,
            lazy_loading=True,
        )
        assert results.shot_histories == {}
        assert len(results.shot_wall_clock_times) == 5

    def test_resume_seed_mismatch_raises_without_decoding_a_shot(
        self, tmp_path, monkeypatch
    ):
        """The resume config check still compares `default_base_seed`, and
        reads no shot to do it."""
        program = _build_seeded_counter_program()
        program.run(num_shots=3, checkpoint=True, checkpoint_dir=tmp_path)
        other = _build_seeded_counter_program()
        other.default_base_seed = 7

        _forbid_shot_history_decode(monkeypatch)
        with pytest.raises(ValueError):
            other.run(
                num_shots=3,
                checkpoint=True,
                resume=True,
                checkpoint_dir=tmp_path,
            )

    def test_loky_executor_checkpointed_run_reads_back_from_store(
        self, tmp_path
    ):
        """Workers return receipts, so a non-lazy parallel run reads every
        history back from the store, and a custom `results_filename` leaves
        one file."""
        loky = pytest.importorskip("loky")
        executor = loky.get_reusable_executor(max_workers=2)
        results = _build_seeded_counter_program().run(
            num_shots=4,
            checkpoint=True,
            checkpoint_dir=tmp_path,
            shot_executor=executor,
            n_shot_batches=2,
            lazy_loading=False,
            results_filename="custom.h5",
            verbose=False,
        )
        assert sorted(results.shot_histories) == [0, 1, 2, 3]
        for i in range(4):
            assert results.shot_histories[i][-1]["counter"] == i + 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["custom.h5"]

    def test_resume_over_a_corrupt_payload_raises_before_any_shot(
        self, tmp_path, monkeypatch
    ):
        """Resume lists the stored keys strictly, so a truncated payload
        raises before any shot is computed."""
        program = _build_seeded_counter_program()
        program.run(num_shots=2, checkpoint=True, checkpoint_dir=tmp_path)
        extra = ProgramResults(lazy_loading=False)
        history = History()
        history.append(Frame({"counter": 3}))
        extra.add_shot(2, history)
        receipt = extra.checkpoint(checkpoint_dir=tmp_path, worker_id="w")
        assert receipt.path is not None
        payload = Path(receipt.path)
        data = payload.read_bytes()
        payload.write_bytes(data[: len(data) // 2])

        def no_shot(*args, **kwargs):
            raise AssertionError("a shot ran")

        monkeypatch.setattr(QuantumProgram, "_run_shot", staticmethod(no_shot))
        with pytest.raises(CheckpointReadError):
            program.run(
                num_shots=4,
                checkpoint=True,
                resume=True,
                force_resume=True,
                checkpoint_dir=tmp_path,
            )

    def test_non_lazy_resume_loads_stored_shots_from_the_store(self, tmp_path):
        """Resuming with lazy_loading=False brings the stored histories into
        memory and computes only the missing shots."""
        program = _build_seeded_counter_program()
        program.run(num_shots=3, checkpoint=True, checkpoint_dir=tmp_path)
        # A resume with more shots needs force_resume (num_shots differs).
        results = program.run(
            num_shots=5,
            checkpoint=True,
            resume=True,
            force_resume=True,
            checkpoint_dir=tmp_path,
            lazy_loading=False,
        )
        assert sorted(results.shot_histories) == [0, 1, 2, 3, 4]
        for i in range(5):
            assert results.shot_histories[i][-1]["counter"] == i + 1


class TestResolveShotBatching:
    """Unit tests for QuantumProgram._resolve_shot_batching."""

    _UNCHECKED = object()

    def _build_simple_program(self):
        """Build a minimal program for testing."""
        trivial_code = trivial_codepack.create_qec_code()
        qubits = ["Q0"]
        stack = [
            {
                "instruction": "Init Patch Trivial",
                "new_patch_label": "L0",
                "qubits": qubits,
            },
        ]
        return QuantumProgram(
            stack,
            default_noise_model=trivial_codepack.create_ideal_model(qubits),
            patch_types={"Trivial": trivial_code},
            default_base_seed=0,
            name="Simple test",
        )

    @pytest.mark.parametrize(
        "num_shots, executor, n_shot_batches, checkpointed, batch_size,"
        " expected_batches, expected_size",
        [
            (25, True, None, False, None, 2, _UNCHECKED),
            (1, True, None, False, None, 1, _UNCHECKED),
            (100, False, None, True, None, _UNCHECKED, 5),
            (100, True, 3, True, None, _UNCHECKED, 34),
            (20, False, 4, True, None, None, 5),
            (20, True, None, True, 5, 4, 5),
        ],
    )
    def test_resolve_shot_batching_values(
        self,
        num_shots,
        executor,
        n_shot_batches,
        checkpointed,
        batch_size,
        expected_batches,
        expected_size,
    ):
        (
            n_resolved,
            size_resolved,
        ) = self._build_simple_program()._resolve_shot_batching(
            num_shots=num_shots,
            # The method never calls the executor
            shot_executor=object() if executor else None,
            n_shot_batches=n_shot_batches,
            checkpoint_enabled=checkpointed,
            checkpoint_batch_size=batch_size,
        )
        if expected_batches is not self._UNCHECKED:
            assert n_resolved == expected_batches
        if expected_size is not self._UNCHECKED:
            assert size_resolved == expected_size

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_shot_batches": 5, "checkpoint_batch_size": 15},
            {"checkpoint_batch_size": 0},
            {"n_shot_batches": 0},
        ],
        ids=[
            "both_given",
            "zero_checkpoint_batch_size",
            "zero_n_shot_batches",
        ],
    )
    def test_resolve_shot_batching_invalid_raises(self, kwargs):
        call_kwargs = {
            "num_shots": 100,
            "shot_executor": object(),
            "n_shot_batches": None,
            "checkpoint_enabled": True,
            "checkpoint_batch_size": None,
            **kwargs,
        }
        with pytest.raises(ValueError):
            self._build_simple_program()._resolve_shot_batching(**call_kwargs)
