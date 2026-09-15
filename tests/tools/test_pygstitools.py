"""Tester for loqs.tools.pygstitools"""

import copy
import functools
import gc
import multiprocessing as mp
import re
import sys
import time
import weakref

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from pygsti.circuits import Circuit
from pygsti.models import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ExperimentDesign

from loqs.backends import DictNoiseModel, STIMQuantumState, StimCircuitGateRep
from loqs.core import PatchGeometry, ProgramResults, QuantumProgram
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as steane_codepack
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.tools import pygstitools
from loqs.tools.paralleltools import ParallelStrategy
from loqs.tools.pygstitools import (
    EdesignRunner,
)

from _shared_checkpoint_test_helpers import (
    _build_shot_executor,
    _crash_once_and_log_shots,
    _wait_for_index_checkpointed,
)


class _TrivialCounterSetup:
    """A minimal single-qubit edesign (an empty circuit and a one-`Gxpi2`
    circuit) plus a `physical_to_logical` mapping onto the trivial-counter
    codepack, shared by every `EdesignRunner` test below.
    The empty circuit never increments the counter (outcome `"0"`); the
    one-gate circuit increments it once (outcome `"1"`).
    """

    def __init__(self):
        pspec = QubitProcessorSpec(
            num_qubits=1, gate_names=["Gxpi2"], qubit_labels=["Q0"]
        )
        self.model = create_explicit_model(pspec, ideal_gate_type="full TP")
        self.circs = [
            Circuit([], line_labels=["Q0"]),
            Circuit([("Gxpi2", "Q0")], line_labels=["Q0"]),
        ]
        self.edesign = ExperimentDesign(self.circs)
        self.physical_to_logical = {
            "rho0": [
                {
                    "instruction": "Init Patch Trivial",
                    "new_patch_label": "L0",
                    "qubits": ["Q0"],
                },
                {
                    "instruction": "Init Counter",
                    "patch_label": "L0",
                    "initial_value": 0,
                },
            ],
            ("Gxpi2", "Q0"): [("Increment", "L0")],
            "Mdefault": [],
        }
        trivial_code = trivial_codepack.create_qec_code()
        ideal_model = trivial_codepack.create_ideal_model(["Q0"])
        self.program_kwargs = dict(
            default_noise_model=ideal_model,
            patch_types={"Trivial": trivial_code},
        )

    def simulate(self, ckpt=None, resume=None, **overrides):
        """Construct and run an `EdesignRunner` with this setup's edesign/
        model/physical_to_logical/`collect_shot_data_args=("counter", -1)`/
        `num_shots=1` as defaults, overridable via `overrides`. `ckpt`, if
        given, both sets `item_checkpoint_dir` and enables checkpointing;
        `resume`, if not given explicitly, is inferred as `True` iff `ckpt`
        already exists with content."""
        if resume is None:
            resume = False
            if ckpt is not None and ckpt.exists() and any(ckpt.iterdir()):
                resume = True

        kwargs = dict(
            edesign=self.edesign,
            physical_model=self.model,
            physical_to_logical=self.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=("counter", -1),
            item_checkpoint_dir=ckpt,
            checkpoint=ckpt is not None,
            resume=resume,
            program_kwargs=self.program_kwargs,
        )
        kwargs.update(overrides)
        return EdesignRunner(**kwargs).run()


@pytest.fixture
def trivial_counter_setup():
    return _TrivialCounterSetup()


class TestPipelineWithMultiplePatches:
    """`EdesignRunner` against a real two-patch [[7,1,3]] program, confirming
    `frame_filter` picks out each patch's own `"FT Logical Z Measure"` output
    correctly regardless of the composite instruction's internal frame count."""

    @staticmethod
    def _steane_qubits(suffix: str) -> list[str]:
        base = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]
        return [f"{q}{suffix}" for q in base]

    def test_two_patch_measurements_combine_into_the_right_joint_outcome(
        self,
    ):
        q0 = self._steane_qubits("_0")
        q1 = self._steane_qubits("_1")
        all_qubits = q0 + q1
        geometry = PatchGeometry(
            patches={"L0": ("L0", q0), "L1": ("L1", q1)}, layout="7_1_3"
        )

        # L0 prepped to |1>_L, L1 left at |0>_L, each independently
        # FT-measured -- no CX, so the expected joint outcome is fixed.
        phys_qubits = ["Q0", "Q1"]
        physical_to_logical = {
            "rho0": [
                {
                    "instruction": "Init State",
                    "state": len(all_qubits),
                    "qubit_labels": all_qubits,
                },
                *geometry.init_patch_entries("Steane"),
                ("FT Zero Prep", "L0"),
                ("X", "L0"),
                ("FT Zero Prep", "L1"),
            ],
            "Mdefault": [
                ("FT Logical Z Measure", "L0"),
                ("FT Logical Z Measure", "L1"),
            ],
        }

        # A minimal 2-qubit model only drives circuit completion; the
        # actual program below runs the full two-patch Steane setup.
        pspec = QubitProcessorSpec(
            num_qubits=len(phys_qubits),
            gate_names=["Gi"],
            qubit_labels=phys_qubits,
            availability={"Gi": [(q,) for q in phys_qubits]},
        )
        physical_model = create_explicit_model(pspec, ideal_gate_type="full unitary")
        circ = Circuit([], line_labels=phys_qubits)
        edesign = ExperimentDesign([circ])

        code = steane_codepack.create_qec_code()
        noise_model = steane_codepack.create_ideal_model(
            all_qubits,
            gaterep=StimCircuitGateRep,
            model_backend=DictNoiseModel,
        )

        runner = EdesignRunner(
            edesign=edesign,
            physical_model=physical_model,
            physical_to_logical=physical_to_logical,
            num_shots=1,
            collect_shot_data_args=[
                {
                    "key": "logical_measurement",
                    "frame_filter": {"patch_label": "L0"},
                },
                {
                    "key": "logical_measurement",
                    "frame_filter": {"patch_label": "L1"},
                },
            ],
            program_kwargs=dict(
                default_noise_model=noise_model,
                state_type=STIMQuantumState,
                patch_types={"Steane": code},
            ),
        )
        ds = runner.run()

        # Deterministic, noiseless model: L0 always "1", L1 always "0".
        assert ds[circ].counts[("10",)] == 1


class TestSimulateDatasetForEdesign:

    def test_end_to_end_counts(self, trivial_counter_setup):
        """A normal (non-checkpointed) run produces the expected per-circuit
        counts."""
        s = trivial_counter_setup
        ds = s.simulate()

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_max_frame_limit_defaults_to_100(self, trivial_counter_setup):
        """With no override, each circuit's program.run still sees
        QuantumProgram.run's own default of 100, unchanged."""
        s = trivial_counter_setup
        seen = []
        real_run = QuantumProgram.run

        def spy_run(self, *args, **kwargs):
            seen.append(kwargs.get("max_frame_limit"))
            return real_run(self, *args, **kwargs)

        QuantumProgram.run = spy_run
        try:
            s.simulate()
        finally:
            QuantumProgram.run = real_run

        assert seen == [100, 100]

    def test_max_frame_limit_is_forwarded_to_program_run(
        self, trivial_counter_setup
    ):
        """An explicit max_frame_limit override reaches every circuit's
        program.run call, not just the default."""
        s = trivial_counter_setup
        seen = []
        real_run = QuantumProgram.run

        def spy_run(self, *args, **kwargs):
            seen.append(kwargs.get("max_frame_limit"))
            return real_run(self, *args, **kwargs)

        QuantumProgram.run = spy_run
        try:
            s.simulate(max_frame_limit=250)
        finally:
            QuantumProgram.run = real_run

        assert seen == [250, 250]


class TestSimulateDatasetForEdesignCheckpointing:

    def test_existing_checkpoint_with_matching_config_auto_resumes(
        self, trivial_counter_setup, tmp_path
    ):
        """Whether a call continues a prior checkpoint is inferred purely
        from item_checkpoint_dir's own on-disk state and a config match --
        there's no separate flag a caller needs to pass."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        s.simulate(ckpt=ckpt)

        ds = s.simulate(ckpt=ckpt)

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_resume_skips_already_checkpointed_circuits(
        self, trivial_counter_setup, tmp_path
    ):
        """Resuming only re-simulates circuits missing from the checkpoint,
        while still returning a complete DataSet covering every circuit."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"

        # Checkpoint only the first circuit up front.
        partial_edesign = ExperimentDesign([s.circs[0]])
        s.simulate(ckpt=ckpt, edesign=partial_edesign)

        # Spy on _run_one_circuit to confirm only the missing circuit is
        # actually recomputed on resume.
        recorded_indices = []
        original_run_one_circuit = pygstitools._run_one_circuit

        def spy_run_one_circuit(circ, index, **kwargs):
            recorded_indices.append(index)
            return original_run_one_circuit(circ, index, **kwargs)

        pygstitools._run_one_circuit = spy_run_one_circuit
        try:
            ds = s.simulate(ckpt=ckpt)
        finally:
            pygstitools._run_one_circuit = original_run_one_circuit

        assert recorded_indices == [1]
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_incomplete_item_is_redone_on_resume(
        self, trivial_counter_setup, tmp_path
    ):
        """An index lost from runner.h5 (simulating a crash where a
        completed item's own checkpoint entry never durably landed) is
        redone on resume, not silently treated as already done."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"

        # First run: complete both circuits.
        ds1 = s.simulate(ckpt=ckpt)
        assert ds1[s.circs[0]].counts[("0",)] == 1
        assert ds1[s.circs[1]].counts[("1",)] == 1

        # Simulate a crash by removing circuit 1's entry directly from
        # runner.h5, the canonical durable store, via a read-mutate-write.
        stored = EdesignRunner.read(ckpt / "runner.h5")
        del stored._reduced_results[1]
        stored.write(ckpt / "runner.h5")

        from loqs.tools.multiprogramrunner import _read_done_union

        assert set(_read_done_union(ckpt).keys()) == {0}

        # Resume: should redo only the missing circuit, and produce a
        # complete, correct DataSet again.
        ds2 = s.simulate(ckpt=ckpt)

        assert set(_read_done_union(ckpt).keys()) == {0, 1}
        assert ds2[s.circs[0]].counts[("0",)] == 1
        assert ds2[s.circs[1]].counts[("1",)] == 1
        # dataset.txt should still reflect both circuits.
        assert (ckpt / "dataset.txt").exists()
        from pygsti.io import read_dataset

        persisted_ds = read_dataset(str(ckpt / "dataset.txt"), verbosity=0)
        assert len(persisted_ds) == 2

    def test_genuine_crash_recovery_via_read_and_run(
        self, trivial_counter_setup, tmp_path
    ):
        """Simulate a crash partway through circuit processing, recover via
        EdesignRunner.read(...).run() (a real on-disk decode-then-run
        recovery), not just in-memory instance reuse."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"

        runner1 = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=("counter", -1),
            item_checkpoint_dir=ckpt,
            checkpoint=True,
            program_kwargs=s.program_kwargs,
        )

        # Patch _run_one_circuit to crash once, at index 1.
        original_run_one_circuit = pygstitools._run_one_circuit
        crash_triggered = []

        def crashing_run_one_circuit(circ, index, **kwargs):
            if index == 1 and not crash_triggered:
                crash_triggered.append(True)
                raise RuntimeError("simulated crash")
            return original_run_one_circuit(circ, index, **kwargs)

        pygstitools._run_one_circuit = crashing_run_one_circuit
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                runner1.run()
        finally:
            pygstitools._run_one_circuit = original_run_one_circuit

        from loqs.tools.multiprogramrunner import _read_done_union

        # Only index 0 should be checkpointed so far.
        assert set(_read_done_union(ckpt).keys()) == {0}

        # Recover via a real on-disk decode-then-run, not in-memory reuse.
        runner2 = EdesignRunner.read(ckpt / "runner.h5")
        ds = runner2.run()

        assert set(_read_done_union(ckpt).keys()) == {0, 1}
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_resume_with_equivalent_collect_shot_data_args_succeeds(
        self, trivial_counter_setup, tmp_path
    ):
        """A resumed call with a semantically-equivalent but differently-spelled
        collect_shot_data_args (tuple vs dict form) should succeed without raising
        a spurious mismatch error, since the normalized forms are identical."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        # Run with tuple form
        s.simulate(ckpt=ckpt, collect_shot_data_args=("counter", -1))

        # Resume with dict form (semantically identical, differently spelled)
        ds = s.simulate(ckpt=ckpt, collect_shot_data_args={"key": "counter", "indices": -1})

        # Verify both circuits' results are correct
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_resume_mismatched_physical_to_logical_raises(
        self, trivial_counter_setup, tmp_path
    ):
        """A resumed call with a different physical_to_logical than the
        checkpoint was written with is a hard error naming that field, even
        though only one leaf value actually changed."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        s.simulate(ckpt=ckpt)

        changed_p2l = copy.deepcopy(s.physical_to_logical)
        changed_p2l["rho0"][1]["initial_value"] = 1

        with pytest.raises(ValueError, match="physical_to_logical"):
            s.simulate(ckpt=ckpt, physical_to_logical=changed_p2l)

    def test_edesign_roundtrip_without_checkpoint_dir(
        self, trivial_counter_setup, tmp_path
    ):
        """Writing and reading an EdesignRunner with checkpoint=False
        (item_checkpoint_dir=None) preserves edesign as bytes tar archive,
        not just None. The round-tripped runner's edesign is equivalent to
        the original, and calling .run() on it succeeds."""
        s = trivial_counter_setup
        h5_path = tmp_path / "runner.h5"

        # Construct and write runner with checkpoint=False (no checkpoint dir)
        runner1 = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=("counter", -1),
            item_checkpoint_dir=None,
            checkpoint=False,
            program_kwargs=s.program_kwargs,
        )
        runner1.write(h5_path)

        # Read it back
        runner2 = EdesignRunner.read(h5_path)

        # Assert edesign is not None and is equivalent
        assert runner2.edesign is not None
        assert (
            set(runner2.edesign.all_circuits_needing_data)
            == set(s.edesign.all_circuits_needing_data)
        )

        # Assert calling .run() on the round-tripped runner succeeds
        ds = runner2.run()
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_edesign_roundtrip_with_checkpoint_dir_still_works(
        self, trivial_counter_setup, tmp_path
    ):
        """Writing and reading an EdesignRunner with checkpoint=True and
        item_checkpoint_dir set still works (uses directory path, not tar)."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        h5_path = tmp_path / "runner.h5"

        # Construct and write runner with checkpoint=True and checkpoint dir
        runner1 = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=("counter", -1),
            item_checkpoint_dir=ckpt,
            checkpoint=True,
            program_kwargs=s.program_kwargs,
        )
        runner1.write(h5_path)

        # Read it back
        runner2 = EdesignRunner.read(h5_path)

        # Assert edesign is not None and is equivalent
        assert runner2.edesign is not None
        assert (
            set(runner2.edesign.all_circuits_needing_data)
            == set(s.edesign.all_circuits_needing_data)
        )

    def test_resume_mismatched_keep_shot_results_raises(
        self, trivial_counter_setup, tmp_path
    ):
        """A resumed call with a different keep_shot_results than the
        checkpoint was written with is a hard error naming that field."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"

        # First run with keep_shot_results=False
        s.simulate(
            ckpt=ckpt,
            keep_shot_results=False,
            shot_checkpoint=False,
        )

        # Resume with keep_shot_results=True (and required shot_checkpoint)
        with pytest.raises(ValueError, match="keep_shot_results"):
            s.simulate(
                ckpt=ckpt,
                keep_shot_results=True,
                shot_checkpoint=True,
                shot_checkpoint_dir=shot_ckpt,
            )

    def test_resume_mismatched_keep_shot_results_force_resume_works(
        self, trivial_counter_setup, tmp_path
    ):
        """force_resume=True bypasses a keep_shot_results mismatch."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"

        # First run with keep_shot_results=False
        s.simulate(
            ckpt=ckpt,
            keep_shot_results=False,
            shot_checkpoint=False,
        )

        # Resume with keep_shot_results=True but force_resume=True
        ds = s.simulate(
            ckpt=ckpt,
            keep_shot_results=True,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt,
            force_resume=True,
        )

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1


class TestSimulateDatasetForEdesignParallel:
    """`EdesignRunner`'s `parallel_strategy` (a
    [](api:ParallelStrategy)) path, against real `loky` and `submitit`
    executors -- both must produce the same `DataSet` a serial run does,
    including through checkpoint/resume and hybrid shot-/program-level
    parallelism together."""

    def test_loky_program_executor_matches_serial_result(
        self, trivial_counter_setup
    ):
        loky = pytest.importorskip("loky")
        s = trivial_counter_setup
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        ds = s.simulate(parallel_strategy=strategy)

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=(
            "submitit unconditionally registers a SIGCONT handler for "
            "every job it runs (submitit/core/job_environment.py), a "
            "POSIX-only signal that doesn't exist in Windows's `signal` "
            "module at all -- a real, unconditional upstream limitation "
            "(submitit targets SLURM, a Linux-only scheduler), not "
            "something fixable from LoQS's side."
        ),
    )
    def test_submitit_program_executor_matches_serial_result(
        self, trivial_counter_setup, tmp_path
    ):
        submitit = pytest.importorskip("submitit")
        s = trivial_counter_setup
        strategy = ParallelStrategy(
            program_executor=submitit.AutoExecutor(
                folder=tmp_path, cluster="local"
            ),
            n_program_chunks=2,
        )

        ds = s.simulate(parallel_strategy=strategy)

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_hybrid_program_and_shot_executor_matches_serial_result(
        self, trivial_counter_setup
    ):
        """program_executor (across circuits) and shot_executor (within
        each circuit's own shots) nested together -- the real hybrid
        parallelism this stage adds, replacing the old guardrail that
        just rejected this combination."""
        loky = pytest.importorskip("loky")
        s = trivial_counter_setup
        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
            shot_executor=_build_shot_executor,
        )

        ds = s.simulate(parallel_strategy=strategy, num_shots=3)

        assert ds[s.circs[0]].counts[("0",)] == 3
        assert ds[s.circs[1]].counts[("1",)] == 3

    def test_loky_program_executor_resume_only_recomputes_missing_circuits(
        self, trivial_counter_setup, tmp_path
    ):
        """A resumed parallel run only re-simulates circuits missing from
        the checkpoint, chunking and dispatching just those, and still
        returns a complete DataSet covering every circuit."""
        loky = pytest.importorskip("loky")
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"

        partial_edesign = ExperimentDesign([s.circs[0]])
        s.simulate(ckpt=ckpt, edesign=partial_edesign)

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
        )

        # Spy on _run_one_circuit via a Manager list, since the real worker
        # processes wouldn't be observable through a plain in-process list.
        manager = mp.Manager()
        recorded_indices = manager.list()
        original_run_one_circuit = pygstitools._run_one_circuit

        def spy_run_one_circuit(circ, index, **kwargs):
            recorded_indices.append(index)
            return original_run_one_circuit(circ, index, **kwargs)

        pygstitools._run_one_circuit = spy_run_one_circuit
        try:
            ds = s.simulate(ckpt=ckpt, parallel_strategy=strategy)
        finally:
            pygstitools._run_one_circuit = original_run_one_circuit

        assert list(recorded_indices) == [1]
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1
        # Verify both circuits are persisted in the dataset checkpoint file
        assert (ckpt / "dataset.txt").exists()
        from pygsti.io import read_dataset
        persisted_ds = read_dataset(str(ckpt / "dataset.txt"), verbosity=0)
        assert len(persisted_ds) == 2
        assert s.circs[0] in persisted_ds
        assert s.circs[1] in persisted_ds


class TestSimulateDatasetForEdesignMemoryBound:

    def test_previous_circuits_program_is_dropped_before_the_next_is_built(
        self, trivial_counter_setup, monkeypatch
    ):
        """Each circuit's QuantumProgram/ProgramResults is actually
        collectible before the next circuit's program is built, not just
        intended to be -- confirms peak memory stays bounded across
        circuits rather than merely relying on the loop eventually
        finishing."""
        s = trivial_counter_setup
        circs = s.circs + [
            Circuit([("Gxpi2", "Q0"), ("Gxpi2", "Q0")], line_labels=["Q0"])
        ]
        edesign = ExperimentDesign(circs)

        live_refs = []
        original_build = pygstitools._build_program_for_circuit

        def tracking_build(circ, physical_model, label_to_logical, **kwargs):
            gc.collect()
            for ref in live_refs:
                assert ref() is None, (
                    "A previous circuit's QuantumProgram was still alive "
                    "when building the next one."
                )
            program = original_build(
                circ, physical_model, label_to_logical, **kwargs
            )
            live_refs.append(weakref.ref(program))
            return program

        monkeypatch.setattr(
            pygstitools, "_build_program_for_circuit", tracking_build
        )

        s.simulate(edesign=edesign)


class TestSimulateDatasetForEdesignShotCheckpointing:
    """Tests for [](api:QuantumProgram.run)'s per-worker HDF5 shot-level
    checkpointing, threaded through `EdesignRunner` via the
    `shot_checkpoint`, `shot_checkpoint_dir`, and `lazy_loading`
    parameters."""

    def test_shot_checkpoint_without_shot_checkpoint_dir_raises(
        self, trivial_counter_setup
    ):
        """shot_checkpoint=True given without shot_checkpoint_dir is a
        configuration error, not something that's silently ignored."""
        s = trivial_counter_setup
        with pytest.raises(ValueError, match="shot_checkpoint_dir"):
            s.simulate(shot_checkpoint=True)

    def test_serial_shot_checkpoint_creates_per_circuit_subdirs(
        self, trivial_counter_setup, tmp_path
    ):
        """A serial (no parallel) run with shot_checkpoint=True and a real
        shot_checkpoint_dir produces per-circuit subdirectories under it, one
        per distinct circuit in the edesign, each containing checkpoint files
        that can be loaded via ProgramResults.load_checkpoint."""
        s = trivial_counter_setup
        shot_ckpt_dir = tmp_path / "shot_checkpoints"
        shot_ckpt_dir.mkdir()

        ds = s.simulate(
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt_dir,
            lazy_loading=False,  # Keep shots in memory for collection
        )

        # Confirm the data is correct
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

        # Confirm per-circuit subdirs exist and contain checkpoints using the
        # new circ_{index} naming scheme
        subdirs = list(shot_ckpt_dir.iterdir())
        assert len(subdirs) == 2, (
            f"Expected 2 circuit subdirs, got {len(subdirs)}: {subdirs}"
        )
        assert set(d.name for d in subdirs) == {"circ_0", "circ_1"}

        # Verify each circuit's checkpoint subdirectory by index
        for circuit_index in range(len(s.circs)):
            circ_subdir = shot_ckpt_dir / f"circ_{circuit_index}"
            assert circ_subdir.exists(), f"Missing subdir: {circ_subdir}"

            # Confirm the checkpoint file exists and can load the right number of shots
            checkpoint_file = circ_subdir / "results.h5"
            assert checkpoint_file.exists(), f"Missing checkpoint: {checkpoint_file}"

            loaded_results = ProgramResults()
            loaded_results.load_checkpoint(checkpoint_dir=circ_subdir)
            assert len(loaded_results.shot_histories) == 1

    def test_parallel_shot_checkpoint_prevents_circuit_collision(
        self, trivial_counter_setup, tmp_path
    ):
        """A parallel run with parallel.n_program_chunks=1 (one worker processes
        both circuits sequentially) and shot_checkpoint_dir set confirms the
        per-circuit subdirectory scheme actually prevents collisions despite
        sharing one worker/hostname_pid."""
        loky = pytest.importorskip("loky")
        s = trivial_counter_setup
        shot_ckpt_dir = tmp_path / "shot_checkpoints"
        shot_ckpt_dir.mkdir()

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=1),
            n_program_chunks=1,
            shot_executor=_build_shot_executor,
        )

        ds = s.simulate(
            parallel_strategy=strategy,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt_dir,
            lazy_loading=False,  # Keep shots in memory for collection
        )

        # Confirm the data is correct
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

        # Confirm both circuit subdirs exist independently using the new
        # circ_{index} naming scheme where index is the circuit's position
        subdirs = list(shot_ckpt_dir.iterdir())
        assert len(subdirs) == 2
        assert set(d.name for d in subdirs) == {"circ_0", "circ_1"}

        # Verify each circuit's checkpoint subdirectory by index
        for circuit_index in range(len(s.circs)):
            circ_subdir = shot_ckpt_dir / f"circ_{circuit_index}"
            assert circ_subdir.exists(), f"Missing subdir: {circ_subdir}"
            checkpoint_file = circ_subdir / "results.h5"
            assert checkpoint_file.exists(), f"Missing checkpoint: {checkpoint_file}"

            # Confirm the checkpoint can be loaded with the right number of shots
            loaded_results = ProgramResults()
            loaded_results.load_checkpoint(checkpoint_dir=circ_subdir)
            assert len(loaded_results.shot_histories) == 1

    def test_keep_shot_results_end_to_end(self, trivial_counter_setup, tmp_path):
        """EdesignRunner with keep_shot_results=True consolidates per-circuit ProgramResults."""
        s = trivial_counter_setup
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        # Create and run the runner directly
        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=("counter", -1),
            item_checkpoint_dir=item_checkpoint_dir,
            checkpoint=True,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
            program_kwargs=s.program_kwargs,
        )
        ds = runner.run()

        # Verify the dataset is correct
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

        # Verify runner has _program_results populated with ProgramResults
        assert len(runner._program_results) == len(s.circs)
        for circ_index in range(len(s.circs)):
            assert circ_index in runner._program_results
            pr = runner._program_results[circ_index]
            # Verify the per-circuit ProgramResults has the correct number of shots
            assert len(pr.shot_histories) == 1

    def test_reduced_results_uses_dataset_storage_format_end_to_end(
        self, trivial_counter_setup, tmp_path
    ):
        """EdesignRunner (the one runner with a real item_key_fn, which
        triggers a second runner.h5 write for the index_map before any
        item is dispatched) still ends up with _reduced_results in
        'dataset' storage format on disk, not stuck in 'groups' format
        from that second, still-empty write."""
        import h5py
        from loqs.core.programresults import _resolve_checkpoint_object_group

        s = trivial_counter_setup
        item_checkpoint_dir = tmp_path / "item_ckpt"

        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=("counter", -1),
            item_checkpoint_dir=item_checkpoint_dir,
            checkpoint=True,
            lazy_loading=False,
            program_kwargs=s.program_kwargs,
        )
        ds = runner.run()

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

        runner_path = item_checkpoint_dir / runner.runner_filename
        with h5py.File(runner_path, "r") as f:
            group = _resolve_checkpoint_object_group(f)
            storage_format = group["_reduced_results"]["dict"]["keys"][
                "iterable"
            ].attrs.get("storage_format", "groups")
            assert storage_format == "dataset", (
                f"Expected _reduced_results keys to use 'dataset' format "
                f"but got '{storage_format}'"
            )

    def test_custom_results_filename_checkpoint_and_resume(
        self, trivial_counter_setup, tmp_path, monkeypatch
    ):
        """A custom results_filename is correctly threaded through to
        QuantumProgram.run() and used for per-circuit shot-level checkpoints.
        After an injected mid-run crash leaves circuit 1's own shot
        checkpoint partial, a resuming run with the same custom filename
        must detect that prior state and recompute only the single missing
        shot -- proving resume-detection actually consults the custom
        filename rather than the default results.h5."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        custom_results_file = "custom_results.h5"

        # First run (custom results_filename, shot checkpointing) crashes
        # after 3 shots: all of circuit 0's 2, plus 1 of circuit 1's 2.
        compute_count = {"n": 0}
        original_run_shot = QuantumProgram._run_shot

        def _run_shot_with_interrupt(self, max_frame_limit, seed, shot_index):
            compute_count["n"] += 1
            if compute_count["n"] > 3:
                raise RuntimeError("Simulated crash mid-dispatch")
            return original_run_shot(self, max_frame_limit, seed, shot_index)

        with pytest.raises(RuntimeError, match="Simulated crash"):
            monkeypatch.setattr(
                QuantumProgram, "_run_shot", _run_shot_with_interrupt
            )
            EdesignRunner(
                edesign=s.edesign,
                physical_model=s.model,
                physical_to_logical=s.physical_to_logical,
                num_shots=2,
                collect_shot_data_args=("counter", -1),
                item_checkpoint_dir=ckpt,
                checkpoint=True,
                shot_checkpoint=True,
                shot_checkpoint_dir=ckpt / "shots",
                results_filename=custom_results_file,
                program_kwargs=s.program_kwargs,
            ).run()

        circ0_shot_ckpt = ckpt / "shots" / "circ_0"
        assert (circ0_shot_ckpt / custom_results_file).exists()
        assert not (circ0_shot_ckpt / "results.h5").exists()
        circ0_results = ProgramResults(results_filename=custom_results_file)
        circ0_results.load_checkpoint(circ0_shot_ckpt)
        assert len(circ0_results.shot_histories) == 2

        # Verify: circuit 0's own custom-named checkpoint is complete (2
        # shots), circuit 1's is partial (1 shot); default results.h5 is absent.
        circ1_shot_ckpt = ckpt / "shots" / "circ_1"
        assert (circ1_shot_ckpt / custom_results_file).exists()
        assert not (circ1_shot_ckpt / "results.h5").exists()
        circ1_results = ProgramResults(results_filename=custom_results_file)
        circ1_results.load_checkpoint(circ1_shot_ckpt)
        assert len(circ1_results.shot_histories) == 1

        # Resume (crash injection removed) should detect circuit 1's partial
        # state via the custom filename and recompute only its missing shot.
        monkeypatch.undo()
        compute_count_on_resume = {"n": 0}
        original_run_shot_2 = QuantumProgram._run_shot

        def _count_compute_calls_resume(self, max_frame_limit, seed, shot_index):
            compute_count_on_resume["n"] += 1
            return original_run_shot_2(self, max_frame_limit, seed, shot_index)

        monkeypatch.setattr(
            QuantumProgram, "_run_shot", _count_compute_calls_resume
        )

        ds2 = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=2,
            collect_shot_data_args=("counter", -1),
            item_checkpoint_dir=ckpt,
            checkpoint=True,
            resume=True,
            shot_checkpoint=True,
            shot_checkpoint_dir=ckpt / "shots",
            results_filename=custom_results_file,
            program_kwargs=s.program_kwargs,
        ).run()

        monkeypatch.undo()

        assert ds2[s.circs[0]].counts[("0",)] == 2
        assert ds2[s.circs[1]].counts[("1",)] == 2

        # Only circuit 1's 1 missing shot should be recomputed, not both
        # circuits' shots from scratch.
        assert compute_count_on_resume["n"] == 1

    def test_keep_shot_results_parallel(self, trivial_counter_setup, tmp_path):
        """EdesignRunner with keep_shot_results=True works under parallel dispatch."""
        loky = pytest.importorskip("loky")
        s = trivial_counter_setup
        item_checkpoint_dir = tmp_path / "item_ckpt"
        shot_checkpoint_dir = tmp_path / "shot_ckpt"

        strategy = ParallelStrategy(
            program_executor=loky.get_reusable_executor(max_workers=2),
            n_program_chunks=2,
        )

        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=("counter", -1),
            parallel_strategy=strategy,
            checkpoint=True,
            item_checkpoint_dir=item_checkpoint_dir,
            shot_checkpoint_dir=shot_checkpoint_dir,
            shot_checkpoint=True,
            keep_shot_results=True,
            lazy_loading=False,
            program_kwargs=s.program_kwargs,
        )
        ds = runner.run()

        # Verify it completes with correct data
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

        # Verify _program_results has one entry per circuit
        assert len(runner._program_results) == 2
        for prog_index in range(2):
            assert prog_index in runner._program_results
            pr = runner._program_results[prog_index]
            # Verify correct shot count
            assert len(pr.shot_histories) == 1
