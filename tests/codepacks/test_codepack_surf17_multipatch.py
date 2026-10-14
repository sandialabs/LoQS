"""Tester for multi-patch surface-code machinery built on codepack_surf17_tomita2014.

Phase A: per-patch syndrome-history namespacing and reference-round decoding.
Phase B: transversal logical CNOT.
Phase C: ancilla-mediated joint logical parity measurements.
"""

import pytest

pygsti = pytest.importorskip("pygsti")

from loqs.backends import (
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMQuantumState,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.core import InstructionStack, PatchGeometry, QuantumProgram
from loqs.core.recordables.pauliframe import PauliFrame
from loqs.codepacks import codepack_surf17_tomita2014 as codepack_surf17
from loqs.codepacks import codepack_surf17_multipatch as multipatch

NUM_STIM_SHOTS = 10
# The dense-statevector smoke test below is fully deterministic (Zero/Plus
# Prep product state, noiseless model), so a single shot is sufficient; the
# dense backend is expensive enough that repeated shots would only add cost
# without adding coverage.
NUM_KRAUS_SHOTS = 1


def layout_qubits(layout: str, suffix: str = "") -> list[str]:
    """Qubit labels for a single patch of the given layout, with a suffix."""
    if layout == "surf10":
        base = [f"D{i}" for i in range(9)] + ["A9"]
    elif layout == "surf13":
        base = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 13)]
    elif layout == "surf17":
        base = [f"D{i}" for i in range(9)] + [f"A{i}" for i in range(9, 17)]
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return [f"{q}{suffix}" for q in base]


def make_stim_program(layout, stack, all_qubits, num_qec_rounds=3):
    """Build a STIM-backed QuantumProgram over the given stack."""
    code = codepack_surf17.create_qec_code(
        layout=layout, num_qec_rounds=num_qec_rounds
    )
    model = codepack_surf17.create_ideal_model(
        all_qubits,
        gaterep=StimCircuitGateRep,
        model_backend=DictNoiseModel,
    )
    return QuantumProgram(
        InstructionStack(stack),
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={"SURF": code},
    )


class TestTwoPatchFoundations:
    """Phase A: per-patch tracked syndrome histories + reference rounds."""

    @pytest.mark.parametrize("layout", ["surf10"])
    def test_two_patch_independence(self, layout):
        """Two patches with different prep bases decode independently.

        L0 preps |0> and measures Z; L1 preps |+> and measures X. With a
        single global syndrome-history store shared across patches, L1's
        decode would have seen L0's (random, wrong-basis) syndromes and vice
        versa, giving nondeterministic outcomes. With syndrome history
        tracked on each patch's own `.data`, both are deterministic 0.
        """
        q0 = layout_qubits(layout, "_0")
        q1 = layout_qubits(layout, "_1")
        all_q = q0 + q1

        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "L0",
                "qubits": q0,
            },
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "L1",
                "qubits": q1,
            },
            ("Zero Prep", "L0"),
            ("Plus Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical X Measure", "L1"),
        ]

        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)

        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert len(per_shot) == NUM_STIM_SHOTS
        for shot_outcomes in per_shot:
            # [L0 outcome, L1 outcome]
            assert shot_outcomes == [0, 0]

        # Per-patch syndrome histories, tracked on each patch's own `.data`,
        # hold one entry per round.
        patches_per_shot = results.collect_shot_data("patches", -1)
        for patch_label in ["L0", "L1"]:
            for check_type in ["X", "Z"]:
                hists = [
                    p[patch_label].data[f"syndrome_history_{check_type}"]
                    for p in patches_per_shot
                ]
                assert all(len(h) == 3 for h in hists), (
                    f"{patch_label} syndrome_history_{check_type} should "
                    "hold 3 rounds"
                )

    @pytest.mark.parametrize("layout", ["surf10"])
    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_reference_round_noiseless(self, layout, basis):
        """ "clean_diff" mode is still deterministic when noiseless.

        Prep matches the measurement basis here, so round 0 is actually
        deterministic; "clean_diff" mode (drop round 0 as its own detector,
        diff round 1 against it, no escape edges) is the correct choice.
        """
        qubits = layout_qubits(layout)
        prep = "Zero Prep" if basis == "Z" else "Plus Prep"
        meas = f"FT Logical {basis} Measure"
        ref_kwarg = {f"reference_round_mode_{basis}": "clean_diff"}

        stack = [
            {
                "instruction": "Init State",
                "state": len(qubits),
                "qubit_labels": qubits,
            },
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "L0",
                "qubits": qubits,
            },
            (prep, "L0"),
            ("QEC", "L0"),
            {"instruction": meas, "patch_label": "L0", **ref_kwarg},
        ]

        program = make_stim_program(layout, stack, qubits)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        outcomes = results.collect_shot_data("logical_measurement", -1)
        assert outcomes == [0] * NUM_STIM_SHOTS


class TestPairwiseCnotFrameConjugation:
    """Phase B: pure-Python Pauli-frame conjugation through pairwise CNOTs."""

    # (ctrl_in, tgt_in) -> (ctrl_out, tgt_out) under CNOT conjugation
    TRUTH_TABLE = [
        ("I", "I", "I", "I"),
        ("X", "I", "X", "X"),
        ("I", "Z", "Z", "Z"),
        ("Z", "I", "Z", "I"),
        ("I", "X", "I", "X"),
        ("Y", "I", "Y", "X"),
        ("I", "Y", "Z", "Y"),
        ("X", "X", "X", "I"),
        ("Z", "Z", "I", "Z"),
        ("Y", "Y", "X", "Z"),
    ]

    @pytest.mark.parametrize("pc,pt,exp_c,exp_t", TRUTH_TABLE)
    def test_single_pair_truth_table(self, pc, pt, exp_c, exp_t):
        frame_c = PauliFrame(["C0"], [pc])
        frame_t = PauliFrame(["T0"], [pt])
        new_c, new_t = multipatch.pairwise_cnot_pauli_frames(
            frame_c, frame_t, ["C0"], ["T0"]
        )
        assert new_c.pauli_frame == [exp_c]
        assert new_t.pauli_frame == [exp_t]
        # Inputs are not mutated
        assert frame_c.pauli_frame == [pc]
        assert frame_t.pauli_frame == [pt]

    def test_multi_pair(self):
        """Pairs act independently; unpaired qubits are untouched."""
        frame_c = PauliFrame(["C0", "C1", "C2", "C3"], ["X", "I", "Y", "Z"])
        frame_t = PauliFrame(["T0", "T1", "T2", "T3"], ["I", "Z", "Y", "X"])
        new_c, new_t = multipatch.pairwise_cnot_pauli_frames(
            frame_c, frame_t, ["C0", "C1", "C2"], ["T0", "T1", "T2"]
        )
        # (X,I)->(X,X); (I,Z)->(Z,Z); (Y,Y)->(X,Z); pair 3 untouched
        assert new_c.pauli_frame == ["X", "Z", "X", "Z"]
        assert new_t.pauli_frame == ["X", "Z", "Z", "X"]


def two_patch_cnot_stack(
    layout, prep_ctrl, prep_tgt, meas, meas_kwargs, after_prep=()
):
    """Stack: preps, QEC, transversal CNOT L0->L1, QEC, FT measures."""
    q0 = layout_qubits(layout, "_0")
    q1 = layout_qubits(layout, "_1")
    all_q = q0 + q1
    geometry = PatchGeometry(
        patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout=layout
    )
    cnot = multipatch.build_transversal_cnot_instruction(geometry)
    stack = [
        {
            "instruction": "Init State",
            "state": len(all_q),
            "qubit_labels": all_q,
        },
        *geometry.init_patch_entries("SURF"),
        (prep_ctrl, "L0"),
        (prep_tgt, "L1"),
        *after_prep,
        ("QEC", "L0"),
        ("QEC", "L1"),
        (cnot, None),
        ("QEC", "L0"),
        ("QEC", "L1"),
        {"instruction": meas, "patch_label": "L0", **meas_kwargs},
        {"instruction": meas, "patch_label": "L1", **meas_kwargs},
    ]
    return stack, all_q


class TestTransversalCnot:
    """Phase B: transversal logical CNOT (copies X_L to the target)."""

    @pytest.mark.parametrize("layout", ["surf10"])
    def test_x_on_control(self, layout):
        """X_L|0>|0> -> CNOT -> |1>|1>: X_L copies to the target patch."""
        stack, all_q = two_patch_cnot_stack(
            layout,
            "Zero Prep",
            "Zero Prep",
            "FT Logical Z Measure",
            {},
            after_prep=(("X", "L0"),),
        )
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert per_shot == [[1, 1]] * NUM_STIM_SHOTS


def bell_joint_parity_stack(layout, ancilla="Qanc"):
    """Bell prep (Plus L0, Zero L1, transversal CNOT) + joint ZZ + XX parity."""
    q0 = layout_qubits(layout, "_0")
    q1 = layout_qubits(layout, "_1")
    all_q = q0 + q1 + [ancilla]
    # Two PatchGeometry objects over the same L0/L1 patches: CNOT uses the
    # directional "ctrl"/"tgt" roles, joint parity the symmetric "a"/"b".
    cnot_geometry = PatchGeometry(
        patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout=layout
    )
    parity_geometry = PatchGeometry(
        patches={"a": ("L0", q0), "b": ("L1", q1)}, layout=layout
    )
    cnot = multipatch.build_transversal_cnot_instruction(cnot_geometry)
    zz = multipatch.build_joint_parity_zz_instruction(parity_geometry, ancilla)
    xx = multipatch.build_joint_parity_xx_instruction(parity_geometry, ancilla)
    stack = [
        {
            "instruction": "Init State",
            "state": len(all_q),
            "qubit_labels": all_q,
        },
        *cnot_geometry.init_patch_entries("SURF"),
        ("Plus Prep", "L0"),
        ("Zero Prep", "L1"),
        ("QEC", "L0"),
        ("QEC", "L1"),
        (cnot, None),
        ("QEC", "L0"),
        ("QEC", "L1"),
        (zz, None),
        (xx, None),
    ]
    return stack, all_q


class TestJointParity:
    """Phase C: ancilla-mediated joint ZZ / XX logical parity measurements."""

    @pytest.mark.parametrize("layout", ["surf10"])
    def test_joint_parities_on_bell(self, layout):
        """A Bell pair has even ZZ AND even XX parity in the same shot.

        Both parities are measured with the SAME ancilla (Imrz resets it),
        proving the ZZ measurement leaves the state intact for XX.
        """
        stack, all_q = bell_joint_parity_stack(layout)
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        zz = results.collect_shot_data(
            "joint_parity_zz_L0_L1", "all", strip_none_entries=True
        )
        xx = results.collect_shot_data(
            "joint_parity_xx_L0_L1", "all", strip_none_entries=True
        )
        assert zz == [[0]] * NUM_STIM_SHOTS
        assert xx == [[0]] * NUM_STIM_SHOTS


class TestBackendFeasibilityWarning:
    """Phase C: clear warning for dense backends at infeasible qubit counts."""

    def test_infeasible_hard_warning(self):
        with pytest.warns(UserWarning):
            n = multipatch.warn_if_backend_infeasible("surf17", "kraus")
        assert n == 34

    def test_borderline_caution_warning(self):
        with pytest.warns(UserWarning):
            n = multipatch.warn_if_backend_infeasible(
                "surf13", "statevector", extra_qubits=1
            )
        assert n == 27


class TestDenseBackendSmoke:
    """Phase C: statevector smoke test on the intended dense configuration."""

    @pytest.mark.slow
    def test_bell_joint_parities_statevector(self):
        """2x surf10 + 1 ancilla (21 qubits), Bell + joint ZZ/XX parities."""
        layout = "surf10"
        stack, all_q = bell_joint_parity_stack(layout)
        code = codepack_surf17.create_qec_code(layout=layout, num_qec_rounds=2)
        model = codepack_surf17.create_ideal_model(
            all_q,
            gaterep=UnitaryGateRep,
            model_backend=DictNoiseModel,
        )
        program = QuantumProgram(
            stack,
            default_noise_model=model,
            state_type=NumpyStatevectorQuantumState,
            patch_types={"SURF": code},
        )
        results = program.run(num_shots=NUM_KRAUS_SHOTS, verbose=False)
        zz = results.collect_shot_data(
            "joint_parity_zz_L0_L1", "all", strip_none_entries=True
        )
        xx = results.collect_shot_data(
            "joint_parity_xx_L0_L1", "all", strip_none_entries=True
        )
        assert zz == [[0]] * NUM_KRAUS_SHOTS
        assert xx == [[0]] * NUM_KRAUS_SHOTS
