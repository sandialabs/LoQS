"""Tester for loqs.tools.pygstitools"""

import copy
import gc
import re
import sys
import weakref

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from pygsti.circuits import Circuit
from pygsti.models import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ExperimentDesign

from loqs.backends import DictNoiseModel, STIMQuantumState, StimCircuitGateRep
from loqs.core import Frame, History, PatchGeometry, ProgramResults, QuantumProgram
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as steane_codepack
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.tools import pygstitools
from loqs.tools.paralleltools import ParallelStrategy
from loqs.tools.pygstitools import (
    convert_edesign_to_programs,
    convert_run_programs_to_dataset,
    EdesignRunner,
)


def _build_shot_executor():
    """Module-level factory (not a closure) building a fresh loky
    executor -- a picklable `shot_executor` factory for hybrid
    shot-/program-level parallelism tests."""
    import loky

    return loky.get_reusable_executor(max_workers=1)


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


def _fake_program(circuit_repr, shot_frames):
    """A minimal stand-in for a QuantumProgram: only the surface
    `convert_run_programs_to_dataset` actually touches (`.name` plus a
    `.run()` returning canned results), skipping a full QuantumProgram/
    codepack setup entirely. `circuit_repr` is the pyGSTi `Circuit`'s own
    `repr()`; `shot_frames` is one list of `Frame` objects per shot, each
    appended in order to that shot's `History`."""

    class _FakeProgram:
        def run(self, *args, **kwargs):
            return self._canned_results

    program = _FakeProgram()
    program.name = circuit_repr
    shot_histories = {
        i: History(frames) for i, frames in enumerate(shot_frames)
    }
    program._canned_results = ProgramResults(shot_histories=shot_histories)
    return program


class TestConvertRunProgramsToDataset:

    def test_tuple_args_give_one_outcome_per_shot(self):
        """A plain tuple-shaped collect_shot_data_args (the single-key
        case) produces exactly one outcome label per shot."""
        circ = Circuit([("Gh", "Q0")], line_labels=["Q0"])
        shots = [
            [Frame({"logical_measurement": 0})],
            [Frame({"logical_measurement": 1})],
            [Frame({"logical_measurement": 1})],
        ]
        program = _fake_program(repr(circ), shots)

        ds = convert_run_programs_to_dataset([program])

        counts = ds[circ].counts
        assert counts[("0",)] == 1
        assert counts[("1",)] == 2

    def test_list_args_join_per_collector_outcomes_in_order(self):
        """A list of per-collector (key, index) args -- e.g. one per logical
        patch -- joins each shot's per-collector values, in list order,
        into a single combined outcome label."""
        circ = Circuit([("Gh", "Q0"), ("Gh", "Q1")], line_labels=["Q0", "Q1"])
        shots = [
            [
                Frame({"logical_measurement": 0, "patch_label": "L0"}),
                Frame({"logical_measurement": 0, "patch_label": "L1"}),
            ],
            [
                Frame({"logical_measurement": 0, "patch_label": "L0"}),
                Frame({"logical_measurement": 1, "patch_label": "L1"}),
            ],
            [
                Frame({"logical_measurement": 1, "patch_label": "L0"}),
                Frame({"logical_measurement": 0, "patch_label": "L1"}),
            ],
            [
                Frame({"logical_measurement": 1, "patch_label": "L0"}),
                Frame({"logical_measurement": 0, "patch_label": "L1"}),
            ],
        ]
        program = _fake_program(repr(circ), shots)

        ds = convert_run_programs_to_dataset(
            [program],
            collect_shot_data_args=[
                ("logical_measurement", -2),
                ("logical_measurement", -1),
            ],
        )

        counts = ds[circ].counts
        assert counts[("00",)] == 1
        assert counts[("01",)] == 1
        assert counts[("10",)] == 2

    def test_auto_runs_a_program_with_no_stored_results(self):
        """A program that hasn't been run yet gets run (at the default
        `num_shots=1`) rather than raising or being skipped."""
        trivial_code = trivial_codepack.create_qec_code()
        ideal_model = trivial_codepack.create_ideal_model(["Q0"])
        stack = [
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
            {
                "instruction": "Increment",
                "patch_label": "L0",
                "increment_by": 2,
            },
        ]
        program = QuantumProgram(
            stack,
            default_noise_model=ideal_model,
            patch_types={"Trivial": trivial_code},
            name="Circuit()",
        )

        ds = convert_run_programs_to_dataset(
            [program], collect_shot_data_args=("counter", -1)
        )

        assert ds[Circuit("()")].counts[("2",)] == 1

    def test_warns_deprecation(self):
        """Calling convert_run_programs_to_dataset warns DeprecationWarning,
        pointing at EdesignRunner as its replacement."""
        circ = Circuit([("Gh", "Q0")], line_labels=["Q0"])
        program = _fake_program(
            repr(circ), [[Frame({"logical_measurement": 0})]]
        )
        with pytest.warns(
            DeprecationWarning, match="EdesignRunner"
        ):
            convert_run_programs_to_dataset([program])


class TestConvertEdesignToPrograms:

    def test_builds_one_program_per_circuit_and_runs_correctly(self):
        """One QuantumProgram per edesign circuit, each running the
        physical_to_logical-mapped instructions for that circuit's gates."""
        pspec = QubitProcessorSpec(
            num_qubits=1, gate_names=["Gxpi2"], qubit_labels=["Q0"]
        )
        model = create_explicit_model(pspec, ideal_gate_type="full TP")
        circs = [
            Circuit([], line_labels=["Q0"]),
            Circuit([("Gxpi2", "Q0")], line_labels=["Q0"]),
        ]
        edesign = ExperimentDesign(circs)

        physical_to_logical = {
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

        programs = convert_edesign_to_programs(
            edesign,
            model,
            physical_to_logical,
            default_noise_model=ideal_model,
            patch_types={"Trivial": trivial_code},
        )

        assert len(programs) == len(edesign.all_circuits_needing_data) == 2

        for program in programs:
            program.run(num_shots=1, verbose=False)
        ds = convert_run_programs_to_dataset(
            programs, collect_shot_data_args=("counter", -1)
        )

        # The empty circuit never increments the counter; the one-gate
        # circuit increments it once.
        assert ds[circs[0]].counts[("0",)] == 1
        assert ds[circs[1]].counts[("1",)] == 1

    def test_warns_deprecation(self, trivial_counter_setup):
        """Calling convert_edesign_to_programs warns DeprecationWarning,
        pointing at EdesignRunner as its replacement."""
        s = trivial_counter_setup
        with pytest.warns(
            DeprecationWarning, match="EdesignRunner"
        ):
            convert_edesign_to_programs(
                s.edesign, s.model, s.physical_to_logical, **s.program_kwargs
            )


class TestPipelineWithMultiplePatches:
    """`convert_edesign_to_programs`/`convert_run_programs_to_dataset`
    against a real two-patch [[7,1,3]] program, confirming `frame_filter`
    picks out each patch's own `"FT Logical Z Measure"` output correctly
    regardless of the composite instruction's internal frame count."""

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
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_qubits),
                "qubit_labels": all_qubits,
            },
            *geometry.init_patch_entries("Steane"),
            ("FT Zero Prep", "L0"),
            ("X", "L0"),
            ("FT Zero Prep", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical Z Measure", "L1"),
        ]
        code = steane_codepack.create_qec_code()
        model = steane_codepack.create_ideal_model(
            all_qubits,
            gaterep=StimCircuitGateRep,
            model_backend=DictNoiseModel,
        )
        program = QuantumProgram(
            stack,
            default_noise_model=model,
            state_type=STIMQuantumState,
            patch_types={"Steane": code},
            name="Circuit()",
        )

        # convert_run_programs_to_dataset always runs each program itself
        # (at its own default num_shots=1), so no need to run() it first.
        ds = convert_run_programs_to_dataset(
            [program],
            collect_shot_data_args=[
                {"key": "logical_measurement", "frame_filter": {"patch_label": "L0"}},
                {"key": "logical_measurement", "frame_filter": {"patch_label": "L1"}},
            ],
        )

        # Deterministic, noiseless model: L0 always "1", L1 always "0".
        assert ds[Circuit("()")].counts[("10",)] == 1


class TestSimulateDatasetForEdesign:

    def test_end_to_end_matches_deprecated_pipeline(self, trivial_counter_setup):
        """A normal (non-checkpointed) run produces the same per-circuit
        counts as the deprecated convert_edesign_to_programs +
        convert_run_programs_to_dataset pipeline."""
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

    def test_checkpoint_run_matches_non_checkpointed_result(
        self, trivial_counter_setup, tmp_path
    ):
        """Checkpointing doesn't change the resulting counts, and leaves a
        checkpoint directory behind."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"

        ds = s.simulate(ckpt=ckpt)

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1
        assert ckpt.exists()
        assert ckpt.is_dir()

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

    def test_existing_content_without_runner_h5_raises(
        self, trivial_counter_setup, tmp_path
    ):
        """Content that isn't a recognized EdesignRunner checkpoint (no
        runner.h5) is never silently overwritten or continued."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        ckpt.mkdir()
        (ckpt / "unrelated.txt").write_text("not a checkpoint")

        with pytest.raises(FileExistsError, match=re.escape(str(ckpt))):
            s.simulate(ckpt=ckpt)

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

        ds = s.simulate(ckpt=ckpt)

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_incomplete_item_is_redone_on_resume(
        self, trivial_counter_setup, tmp_path
    ):
        """A partial run can be resumed, completing only the missing circuit,
        and the final DataSet covers all circuits."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"

        # First partial run: complete only the first circuit
        partial_edesign = ExperimentDesign([s.circs[0]])
        s.simulate(ckpt=ckpt, edesign=partial_edesign)

        # Resume with full edesign: should run only the missing circuit
        ds = s.simulate(ckpt=ckpt)

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1
        # Verify dataset.txt contains both circuits
        assert (ckpt / "dataset.txt").exists()
        from pygsti.io import read_dataset
        persisted_ds = read_dataset(str(ckpt / "dataset.txt"), verbosity=0)
        assert len(persisted_ds) == 2

    def test_resume_mismatched_num_shots_raises(
        self, trivial_counter_setup, tmp_path
    ):
        """A resumed call with a different num_shots than the checkpoint
        was written with is a hard error naming that field."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        s.simulate(ckpt=ckpt)

        with pytest.raises(ValueError, match="num_shots"):
            s.simulate(ckpt=ckpt, num_shots=2)

    def test_resume_mismatched_collect_shot_data_args_raises(
        self, trivial_counter_setup, tmp_path
    ):
        """A resumed call with a different collect_shot_data_args than the
        checkpoint was written with is a hard error naming that field."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        s.simulate(ckpt=ckpt)

        with pytest.raises(ValueError, match="collect_shot_data_args"):
            s.simulate(ckpt=ckpt, collect_shot_data_args=("counter", -2))

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

    def test_force_resume_bypasses_config_mismatch(
        self, trivial_counter_setup, tmp_path
    ):
        """force_resume=True proceeds despite a config mismatch that would
        otherwise raise, rather than requiring an untouched checkpoint."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"
        s.simulate(ckpt=ckpt)

        ds = s.simulate(ckpt=ckpt, force_resume=True, num_shots=2)

        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

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


def _checkpointed_circuits(checkpoint_path) -> set:
    """Every circuit with a row in a checkpoint text file, parsed as
    pyGSTi `Circuit`s -- used to confirm a resumed parallel run actually
    appends the circuits it recomputed, not just returns them in-memory.
    """
    from pygsti.io import read_dataset

    return set(read_dataset(str(checkpoint_path / "dataset.txt"), verbosity=0).keys())


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
        ds = s.simulate(ckpt=ckpt, parallel_strategy=strategy)

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

    def test_resume_cascades_into_item_partial_shot_checkpoint(
        self, trivial_counter_setup, tmp_path, monkeypatch
    ):

        """When an item (circuit) crashes partway through its own shot-level
        checkpoint, a runner-level resume must cascade the resume flag down to
        that item's own QuantumProgram.run() call, causing it to resume from its
        partial shot checkpoint rather than recomputing all shots from scratch.
        This differs from test_incomplete_item_is_redone_on_resume, which only
        covers items that hadn't started shot-level work at all."""
        s = trivial_counter_setup
        item_ckpt = tmp_path / "item_checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"
        item_ckpt.mkdir()
        shot_ckpt.mkdir()

        # First run: simulate a crash partway through the second circuit's
        # own shot work (3 shots total, so we'll interrupt at shot 2)
        compute_count = {"n": 0}
        original_run_shot = QuantumProgram._run_shot

        def _run_shot_with_interrupt(self, max_frame_limit, seed, shot_index):
            compute_count["n"] += 1
            # Crash after 3 shots total (completing all of circuit 0's 1 shot,
            # and 2 of circuit 1's 2 shots)
            if compute_count["n"] > 3:
                raise RuntimeError("Simulated crash mid-dispatch")
            return original_run_shot(self, max_frame_limit, seed, shot_index)

        with pytest.raises(RuntimeError, match="Simulated crash"):
            monkeypatch.setattr(
                QuantumProgram, "_run_shot", _run_shot_with_interrupt
            )
            s.simulate(
                ckpt=item_ckpt,
                shot_checkpoint=True,
                shot_checkpoint_dir=shot_ckpt,
                num_shots=2,  # 2 shots per circuit
                lazy_loading=False,
            )

        # Verify: circuit 0's shot checkpoint should be complete (2 shots)
        circ0_shot_ckpt = shot_ckpt / "circ_0"
        assert circ0_shot_ckpt.exists()
        circ0_results = ProgramResults()
        circ0_results.load_checkpoint(circ0_shot_ckpt)
        assert len(circ0_results.shot_histories) == 2

        # Verify: circuit 1's shot checkpoint should be partial (1 of 2 shots)
        circ1_shot_ckpt = shot_ckpt / "circ_1"
        assert circ1_shot_ckpt.exists()
        circ1_results_partial = ProgramResults()
        circ1_results_partial.load_checkpoint(circ1_shot_ckpt)
        assert len(circ1_results_partial.shot_histories) == 1

        # Second run: item-level resume should cascade down to circuit 1's
        # own QuantumProgram.run() call, resuming its partial checkpoint.
        monkeypatch.undo()
        compute_count_on_resume = {"n": 0}

        original_run_shot_2 = QuantumProgram._run_shot

        def _count_compute_calls_resume(self, max_frame_limit, seed, shot_index):
            compute_count_on_resume["n"] += 1
            return original_run_shot_2(self, max_frame_limit, seed, shot_index)

        monkeypatch.setattr(
            QuantumProgram, "_run_shot", _count_compute_calls_resume
        )

        ds = s.simulate(
            ckpt=item_ckpt,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt,
            num_shots=2,
            lazy_loading=False,
        )

        monkeypatch.undo()

        # Verify: the results are fully correct (all 2 shots for both circuits)
        assert ds[s.circs[0]].counts[("0",)] == 2
        assert ds[s.circs[1]].counts[("1",)] == 2

        # Only circuit 1's 1 missing shot should be recomputed, not both
        # (which would mean it was redone from scratch instead of resumed).
        assert compute_count_on_resume["n"] == 1

    def test_resume_does_not_cascade_raise_for_item_with_no_shot_checkpoint(
        self, trivial_counter_setup, tmp_path, monkeypatch
    ):
        """When an item (circuit) has no shot-level checkpoint results.h5,
        a runner-level resume must not cascade resume=True down to that item's
        QuantumProgram.run() call, avoiding the case (d) ValueError
        ("resume=True with no on-disk state"). Instead, resume=False is passed,
        and the item is redone from scratch and completes successfully."""
        s = trivial_counter_setup
        item_ckpt = tmp_path / "item_checkpoint"
        shot_ckpt = tmp_path / "shot_checkpoint"
        item_ckpt.mkdir()
        shot_ckpt.mkdir()

        # First run: complete circuit 0, then crash on circuit 1's first shot
        # before its checkpoint batch can flush.
        compute_count = {"n": 0}
        original_run_shot = QuantumProgram._run_shot

        def _run_shot_with_interrupt(self, max_frame_limit, seed, shot_index):
            compute_count["n"] += 1
            # Crash at shot 4 (circuit 1's first shot, after circuit 0's 3)
            if compute_count["n"] >= 4:
                raise RuntimeError("Simulated crash mid-circuit-1")
            return original_run_shot(self, max_frame_limit, seed, shot_index)

        with pytest.raises(RuntimeError, match="mid-circuit-1"):
            monkeypatch.setattr(
                QuantumProgram, "_run_shot", _run_shot_with_interrupt
            )
            s.simulate(
                ckpt=item_ckpt,
                shot_checkpoint=True,
                shot_checkpoint_dir=shot_ckpt,
                num_shots=3,
                lazy_loading=False,
            )

        monkeypatch.undo()

        # Manually verify/set up the precondition: circuit 0 complete,
        # circuit 1 subdirectory exists but may or may not have results.h5
        circ0_shot_ckpt = shot_ckpt / "circ_0"
        assert circ0_shot_ckpt.exists()
        circ0_results = ProgramResults()
        circ0_results.load_checkpoint(circ0_shot_ckpt)
        assert len(circ0_results.shot_histories) == 3

        # If circuit 1's results.h5 exists, remove it to simulate the case where
        # circuit 1's batch didn't complete before the crash
        circ1_shot_ckpt = shot_ckpt / "circ_1"
        circ1_results_file = circ1_shot_ckpt / "results.h5"
        if circ1_results_file.exists():
            circ1_results_file.unlink()

        # Ensure the precondition is met: circuit 1 has no results.h5
        assert not circ1_results_file.exists(), (
            "Precondition setup failed: circuit 1 results.h5 should be removed"
        )

        # Second run: resume should complete successfully without raising case (d).
        # The cascading logic checks for results.h5; since it doesn't exist,
        # resume=False is passed to circuit 1's QuantumProgram.run().
        ds = s.simulate(
            ckpt=item_ckpt,
            shot_checkpoint=True,
            shot_checkpoint_dir=shot_ckpt,
            num_shots=3,
            lazy_loading=False,
        )

        # Verify: the results are fully correct (all 3 shots for both circuits)
        assert ds[s.circs[0]].counts[("0",)] == 3
        assert ds[s.circs[1]].counts[("1",)] == 3

        # Verify: circuit 1's shot checkpoint now has results.h5 and is complete
        assert circ1_results_file.exists()
        circ1_results_after = ProgramResults()
        circ1_results_after.load_checkpoint(circ1_shot_ckpt)
        assert len(circ1_results_after.shot_histories) == 3

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
