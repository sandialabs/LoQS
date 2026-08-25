"""Tester for loqs.tools.pygstitools"""

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
from loqs.tools.pygstitools import (
    convert_edesign_to_programs,
    convert_run_programs_to_dataset,
)


def _fake_program(circuit_repr, shot_frames):
    """A minimal stand-in for a QuantumProgram: only the surface
    `convert_run_programs_to_dataset` actually touches (`.name` plus a
    pre-populated `_last_results`), skipping a full QuantumProgram/codepack
    setup entirely.

    Parameters
    ----------
    circuit_repr : str
        `repr()` of the pyGSTi `Circuit` this program corresponds to.
    shot_frames : list[list[Frame]]
        One list of `Frame` objects per shot, appended in order to that
        shot's `History`.
    """

    class _FakeProgram:
        pass

    program = _FakeProgram()
    program.name = circuit_repr
    shot_histories = {
        i: History(frames) for i, frames in enumerate(shot_frames)
    }
    program._last_results = ProgramResults(shot_histories=shot_histories)
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
        """A program with no `_last_results` set gets run (at the default
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
        assert getattr(program, "_last_results", None) is None

        ds = convert_run_programs_to_dataset(
            [program], collect_shot_data_args=("counter", -1)
        )

        assert ds[Circuit("()")].counts[("2",)] == 1


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

        program.run(num_shots=5, verbose=False)
        ds = convert_run_programs_to_dataset(
            [program],
            collect_shot_data_args=[
                {"key": "logical_measurement", "frame_filter": {"patch_label": "L0"}},
                {"key": "logical_measurement", "frame_filter": {"patch_label": "L1"}},
            ],
        )

        # Deterministic, noiseless model: L0 always "1", L1 always "0".
        assert ds[Circuit("()")].counts[("10",)] == 5
