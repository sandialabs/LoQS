"""Tester for loqs.tools.pygstitools"""

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from pygsti.circuits import Circuit
from pygsti.models import create_explicit_model
from pygsti.processors import QubitProcessorSpec
from pygsti.protocols import ExperimentDesign

from loqs.backends import DictNoiseModel, STIMQuantumState, StimCircuitGateRep
from loqs.core import PatchGeometry
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as steane_codepack
from loqs.codepacks import codepack_trivial_counter as trivial_codepack
from loqs.tools.pygstitools import EdesignRunner


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
        model/physical_to_logical/`collect_shot_data_args=[("counter", -1)]`/
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
            collect_shot_data_args=[("counter", -1)],
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
    correctly regardless of the composite instruction's internal frame count.
    """

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
        physical_model = create_explicit_model(
            pspec, ideal_gate_type="full unitary"
        )
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


class TestSimulateDatasetForEdesignCheckpointing:
    """EdesignRunner-specific serialization tests: generic checkpoint/
    resume/mismatch-check machinery is covered generically in
    test_multiprogramrunner.py against test-double runners. These two
    exercise what's genuinely EdesignRunner-specific -- its own pyGSTi
    ExperimentDesign tar/directory serialization."""

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
            collect_shot_data_args=[("counter", -1)],
            item_checkpoint_dir=None,
            checkpoint=False,
            program_kwargs=s.program_kwargs,
        )
        runner1.write(h5_path)

        # Read it back
        runner2 = EdesignRunner.read(h5_path)

        # Assert edesign is not None and is equivalent
        assert runner2.edesign is not None
        assert set(runner2.edesign.all_circuits_needing_data) == set(
            s.edesign.all_circuits_needing_data
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
            collect_shot_data_args=[("counter", -1)],
            item_checkpoint_dir=ckpt,
            checkpoint=True,
            program_kwargs=s.program_kwargs,
        )
        runner1.write(h5_path)

        # Read it back
        runner2 = EdesignRunner.read(h5_path)

        # Assert edesign is not None and is equivalent
        assert runner2.edesign is not None
        assert set(runner2.edesign.all_circuits_needing_data) == set(
            s.edesign.all_circuits_needing_data
        )

    def test_item_key_fn_is_picklable(self, trivial_counter_setup, tmp_path):
        """Regression test: item_key_fn and snapshot bound methods must be picklable.

        Raw pickle-based parallel backends (e.g. mpi4py.futures.MPIPoolExecutor,
        unlike loky/submitit which use cloudpickle) fail on unpicklable local
        lambdas. Confirms both the bare item_key_fn and the MultiProgramRunner
        _static_kwargs-style snapshot-then-pickle path (runner_snapshot.build_program)
        are picklable.
        """
        import pickle

        s = trivial_counter_setup
        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=[("counter", -1)],
            item_checkpoint_dir=tmp_path / "ckpt",
            checkpoint=True,
        )

        # item_key_fn must be picklable (a module-level function, not a local lambda)
        pickle.dumps(runner.item_key_fn)

        import copy

        runner_snapshot = copy.copy(runner)
        pickle.dumps(runner_snapshot.build_program)


class TestEdesignRunnerHooks:
    """Direct unit tests of EdesignRunner's own derived build_program/
    reduce_program_outcomes/_build_output/_mismatch_check_fields hook
    implementations, in isolation from MultiProgramRunner's generic
    dispatch/checkpoint/resume/parallel machinery -- that generic behavior
    is covered once, generically, in test_multiprogramrunner.py."""

    def test_build_program_resolves_correct_circuit_via_scrambled_index_map(
        self, trivial_counter_setup, tmp_path
    ):
        """build_program's index -> circuit reverse lookup must use
        index_map, not treat index as a plain position in self.items:
        scrambling a resumed run's persisted index_map (still a valid
        bijection) must not corrupt which circuit each index's program
        builds."""
        s = trivial_counter_setup
        ckpt = tmp_path / "checkpoint"

        # First run: establishes index_map {circs[0].str: 0, circs[1].str: 1}.
        s.simulate(ckpt=ckpt)

        # Scramble the persisted index_map (still a valid bijection) and drop
        # all reduced results, forcing a full rebuild under the scrambled map.
        stored = EdesignRunner.read(ckpt / "runner.h5")
        stored.index_map = {
            s.circs[0].str: 1,
            s.circs[1].str: 0,
        }
        stored._reduced_results = {}
        stored.write(ckpt / "runner.h5")

        ds = s.simulate(ckpt=ckpt)

        # Regardless of the scrambled index<->circuit assignment, each
        # circuit must still be paired with its own correct outcome.
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1

    def test_build_program_before_run_does_not_cache_stale_positional_map(
        self, trivial_counter_setup, tmp_path
    ):
        """A stray build_program() call before .run() ever executes must not
        permanently cache a positional index<->circuit mapping: once a real,
        non-positional index_map later becomes available, _circuit_for_index
        must rebuild its cache rather than keep serving the stale one."""
        s = trivial_counter_setup
        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=[("counter", -1)],
            checkpoint=True,
            item_checkpoint_dir=tmp_path,
            program_kwargs=s.program_kwargs,
        )

        # Stray call before .run() ever executes: index_map is still None
        # here, so _circuit_for_index falls back to positional caching.
        runner.build_program(0)
        assert runner._circuits_by_index[0] is s.circs[0]

        # Simulate what .run() would later populate: a real, non-positional
        # index_map (still a valid bijection, just not position-matching).
        runner.index_map = {
            s.circs[0].str: 1,
            s.circs[1].str: 0,
        }

        # Without the fix, this returns the stale positionally-cached
        # circs[0] (wrong -- the index_map says index 0 resolves to circs[1]).
        resolved = runner._circuit_for_index(0)
        assert resolved is s.circs[1]

    def test_reduce_program_outcomes_produces_count_dict(
        self, trivial_counter_setup
    ):
        """reduce_program_outcomes turns one program's shot outcomes into
        a {(label,): count} dict via Counter, independent of any
        dispatch/checkpoint machinery."""
        s = trivial_counter_setup
        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=4,
            collect_shot_data_args=[("counter", -1)],
            program_kwargs=s.program_kwargs,
        )
        program = runner.build_program(1)  # the one-gate circuit, outcome "1"
        program_results = program.run(num_shots=4, verbose=False)
        count_dict = runner.reduce_program_outcomes(program_results)
        assert count_dict == {("1",): 4}

    def test_build_output_assembles_dataset_with_provenance_comment(
        self, trivial_counter_setup
    ):
        """_build_output builds a DataSet from (circuit, count_dict) pairs
        and always sets the provenance comment, unconditionally (no
        dependency on checkpointing being enabled)."""
        s = trivial_counter_setup
        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=[("counter", -1)],
            program_kwargs=s.program_kwargs,
        )
        ordered_results = [
            (s.circs[0], {("0",): 1}),
            (s.circs[1], {("1",): 1}),
        ]
        ds = runner._build_output(ordered_results)
        assert ds[s.circs[0]].counts[("0",)] == 1
        assert ds[s.circs[1]].counts[("1",)] == 1
        assert ds.comment

    def test_mismatch_check_fields_lists_expected_fields(
        self, trivial_counter_setup
    ):
        """max_frame_limit is dropped from the mismatch-check list entirely
        now that it lives in the base class's generic run_kwargs, which is
        never itself mismatch-checked (matching NoiseSweepRunner/
        FaultInjectionRunner precedent)."""
        s = trivial_counter_setup
        runner = EdesignRunner(
            edesign=s.edesign,
            physical_model=s.model,
            physical_to_logical=s.physical_to_logical,
            num_shots=1,
            collect_shot_data_args=[("counter", -1)],
            program_kwargs=s.program_kwargs,
        )
        assert runner._mismatch_check_fields() == [
            "num_shots",
            "_normalized_collect_shot_data_args",
            "physical_to_logical",
            "keep_shot_results",
        ]
