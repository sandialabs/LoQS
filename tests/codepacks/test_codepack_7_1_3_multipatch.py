"""Tester for the [[7,1,3]] codepack's transversal logical CX.

No QEC rounds appear anywhere in these stacks -- that is the boundary of
what codepack_7_1_3_multipatch's bookkeeping currently supports (see its
module docstring).
"""

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from loqs.backends import DictNoiseModel, STIMQuantumState, StimCircuitGateRep
from loqs.core import PatchGeometry, QuantumProgram
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as codepack_steane
from loqs.codepacks import codepack_7_1_3_multipatch as multipatch

NUM_SHOTS = 20


def steane_qubits(suffix: str = "") -> list[str]:
    """A [[7,1,3]] patch's (auxiliary + data) qubit labels, with a suffix."""
    base = ["A0", "A1", "A2"] + [f"D{i}" for i in range(7)]
    return [f"{q}{suffix}" for q in base]


def two_patch_cx_stack(prep_ctrl, prep_tgt, meas):
    """Stack: preps, transversal CX L0->L1, FT measures (no QEC rounds)."""
    q0 = steane_qubits("_0")
    q1 = steane_qubits("_1")
    all_q = q0 + q1
    geometry = PatchGeometry(
        patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout="7_1_3"
    )
    cx = multipatch.build_transversal_cx_instruction(geometry)
    stack = [
        {
            "instruction": "Init State",
            "state": len(all_q),
            "qubit_labels": all_q,
        },
        *geometry.init_patch_entries("Steane"),
        (prep_ctrl, "L0"),
        (prep_tgt, "L1"),
        (cx, None),
        (meas, "L0"),
        (meas, "L1"),
    ]
    return stack, all_q


def make_program(stack, all_qubits):
    """Build a STIM-backed QuantumProgram over the given stack."""
    code = codepack_steane.create_qec_code()
    model = codepack_steane.create_ideal_model(
        all_qubits,
        gaterep=StimCircuitGateRep,
        model_backend=DictNoiseModel,
    )
    return QuantumProgram(
        stack,
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={"Steane": code},
        name="Transversal CX",
    )


class TestConjugateCxLogicalPauliFrames:
    """`conjugate_cx_logical_pauli_frames` truth table (frame = [Z_bit, X_bit])."""

    TRUTH_TABLE = [
        # (frame_ctrl, frame_tgt, expected_ctrl, expected_tgt)
        ([0, 0], [0, 0], [0, 0], [0, 0]),
        ([0, 1], [0, 0], [0, 1], [0, 1]),  # ctrl X copies to tgt
        ([1, 0], [0, 0], [1, 0], [0, 0]),  # ctrl Z untouched
        ([0, 0], [1, 0], [1, 0], [1, 0]),  # tgt Z copies to ctrl
        ([0, 0], [0, 1], [0, 0], [0, 1]),  # tgt X untouched
        ([1, 1], [1, 1], [0, 1], [1, 0]),
    ]

    @pytest.mark.parametrize("fc,ft,exp_c,exp_t", TRUTH_TABLE)
    def test_truth_table(self, fc, ft, exp_c, exp_t):
        orig_fc, orig_ft = list(fc), list(ft)
        new_c, new_t = multipatch.conjugate_cx_logical_pauli_frames(fc, ft)
        assert new_c == exp_c
        assert new_t == exp_t
        # Inputs are not mutated
        assert fc == orig_fc
        assert ft == orig_ft


class TestTransversalCx:
    """Truth table and Bell-correlation checks for the transversal logical CX."""

    def test_zero_zero(self):
        """|0>|0> -> CX -> |0>|0>: both Z measures deterministic 0."""
        stack, all_q = two_patch_cx_stack(
            "Non-FT Zero Prep", "Non-FT Zero Prep", "FT Logical Z Measure"
        )
        program = make_program(stack, all_q)
        results = program.run(num_shots=NUM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(l0 == 0 and l1 == 0 for l0, l1 in per_shot)

    def test_x_on_control(self):
        """X_L|0>|0> -> CX -> |1>|1>: X_L copies to the target patch."""
        stack, all_q = two_patch_cx_stack(
            "Non-FT Zero Prep", "Non-FT Zero Prep", "FT Logical Z Measure"
        )
        # Insert an X on the control patch right after its own prep.
        stack.insert(4, ("X", "L0"))
        program = make_program(stack, all_q)
        results = program.run(num_shots=NUM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(l0 == 1 and l1 == 1 for l0, l1 in per_shot)

    def test_bell_correlations(self):
        """|+>|0> -> CX -> Bell pair: per-patch Z outcomes agree every shot."""
        stack, all_q = two_patch_cx_stack(
            "Non-FT Zero Prep", "Non-FT Zero Prep", "FT Logical Z Measure"
        )
        # No dedicated "Non-FT Plus Prep" exists; build |+>_L via H on |0>_L.
        stack.insert(4, ("H", "L0"))
        program = make_program(stack, all_q)
        results = program.run(num_shots=NUM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(l0 == l1 for l0, l1 in per_shot)
