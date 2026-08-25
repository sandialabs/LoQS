"""Tester for the [[7,1,3]] codepack's transversal logical CX."""

import pytest

pygsti = pytest.importorskip("pygsti")
stim = pytest.importorskip("stim")

from loqs.backends import (
    DictNoiseModel,
    PyGSTiPhysicalCircuit,
    STIMQuantumState,
    StimCircuitGateRep,
)
from loqs.core import PatchGeometry, QuantumProgram
from loqs.core.instructions import builders
from loqs.codepacks import codepack_7_1_3_quantinuum2021 as codepack_steane
from loqs.codepacks import codepack_7_1_3_multipatch as multipatch
from loqs.tools import fttools

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


def make_program(stack, all_qubits, code=None):
    """Build a STIM-backed QuantumProgram over the given stack.

    Parameters
    ----------
    code:
        A pre-built QECCode to use in place of a fresh
        `codepack_steane.create_qec_code()`, e.g. one with an
        `"Injected Data Error"` instruction added for fault-injection
        tests.
    """
    if code is None:
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


def code_with_injected_error(error_gate: str, data_qubit: str):
    """A `codepack_steane` QECCode with an `"Injected Data Error"` instruction.

    The instruction applies a single-qubit `error_gate` (e.g. `"Gxpi"`) to
    `data_qubit` (a template label, e.g. `"D6"`), to be invoked as
    `("Injected Data Error", <patch_label>)` in a program stack.
    """
    code = codepack_steane.create_qec_code()
    error_circ = PyGSTiPhysicalCircuit(
        [[(error_gate, data_qubit)]], qubit_labels=steane_qubits()
    )
    code.instructions["Injected Data Error"] = (
        builders.build_physical_circuit_instruction(
            error_circ, name="Injected Data Error"
        )
    )
    return code


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


class TestConjugateCxLatestSyndromes:
    """`conjugate_cx_latest_syndromes` truth table (syndrome = [S1..S6])."""

    TRUTH_TABLE = [
        # (syndrome_ctrl, syndrome_tgt, expected_ctrl, expected_tgt)
        (
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ),
        # ctrl X-type (S1) alone: tgt is all-zero, so nothing changes.
        (
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ),
        # tgt X-type (S1): ctrl's X-type picks it up; tgt's own is untouched.
        (
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ),
        # ctrl Z-type (S4): untouched on ctrl, copies onto tgt's Z-type.
        (
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ),
        # tgt Z-type (S4) alone: ctrl is all-zero, so nothing changes.
        (
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ),
        # Both fully set: X-type slices cancel on ctrl, Z-type cancel on tgt.
        (
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ),
    ]

    @pytest.mark.parametrize("sc,st,exp_c,exp_t", TRUTH_TABLE)
    def test_truth_table(self, sc, st, exp_c, exp_t):
        orig_sc, orig_st = list(sc), list(st)
        new_c, new_t = multipatch.conjugate_cx_latest_syndromes(sc, st)
        assert new_c == exp_c
        assert new_t == exp_t
        # Inputs are not mutated
        assert sc == orig_sc
        assert st == orig_st


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


class TestSyndromeCouplingAfterCx:
    """Regression coverage for the `conjugate_cx_latest_syndromes` fix.

    Unlike `TestTransversalCx`, these stacks run a QEC round on the
    control patch *before* the CX (to establish a genuinely nonzero
    `latest_syndrome` there) and a QEC round on the target patch *after*
    the CX (to exercise the coupling the fix corrects for).
    """

    def _two_patch_stack_with_qec(self, error_gate, error_qubit):
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
            ("Non-FT Zero Prep", "L0"),
            ("Non-FT Zero Prep", "L1"),
            ("Injected Data Error", "L0"),
            ("Adaptive QEC", "L0"),
            (cx, None),
            ("Adaptive QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical Z Measure", "L1"),
        ]
        code = code_with_injected_error(error_gate, error_qubit)
        return stack, all_q, code

    def test_qec_round_after_cx_uses_coupled_syndrome(self):
        """Coupling-induced syndrome shift must not be mistaken for a new error.

        An X-type error on the control's D6 is fully corrected (virtually)
        by a QEC round *before* the CX, leaving a nonzero
        `latest_syndrome[3:]` (Z-type) baseline on the control -- the
        control's data qubits are still physically perturbed, only
        virtually tracked via the Pauli frame. The transversal CX
        physically couples this into the target's own Z-type checks (per
        the standard CSS map). Without `conjugate_cx_latest_syndromes`,
        the target's subsequent QEC round would misread that coupling as
        a brand-new error on the target and wrongly flip its frame; with
        the fix, both patches measure the correct (unperturbed) logical Z
        outcome.
        """
        stack, all_q, code = self._two_patch_stack_with_qec("Gxpi", "D6")
        program = make_program(stack, all_q, code=code)
        results = program.run(num_shots=NUM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(l0 == 0 and l1 == 0 for l0, l1 in per_shot)

    def test_ft_prep_two_qec_rounds_after_cx(self):
        """FT-prep version of the coupled-syndrome test, over two QEC rounds.

        Same scenario as `test_qec_round_after_cx_uses_coupled_syndrome`,
        but using `FT Zero Prep` (the repeat-until-success protocol) for
        both patches instead of `Non-FT Zero Prep`, and two consecutive
        `Adaptive QEC` rounds on the target after the CX rather than one --
        exercising the module docstring's claim that a single
        `latest_syndrome` correction at CX time is sufficient for
        arbitrarily many subsequent rounds, not just one.
        """
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
            ("FT Zero Prep", "L0"),
            ("FT Zero Prep", "L1"),
            ("Injected Data Error", "L0"),
            ("Adaptive QEC", "L0"),
            (cx, None),
            ("Adaptive QEC", "L1"),
            ("Adaptive QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical Z Measure", "L1"),
        ]
        code = code_with_injected_error("Gxpi", "D6")
        program = make_program(stack, all_q, code=code)
        results = program.run(num_shots=NUM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(l0 == 0 and l1 == 0 for l0, l1 in per_shot)


class TestExhaustiveFaultInjectionAcrossCx:
    """Exhaustive single-fault sweep over the CX's own physical circuit.

    Unlike `TestSyndromeCouplingAfterCx`, which injects one representative
    pre-CX data error to exercise the `latest_syndrome` coupling fix, this
    sweeps every possible single-fault location *inside* the transversal
    CX's own circuit (the 7-`Gcnot` layer) using `fttools`: every weight-1
    Pauli fault before each `Gcnot`, and every weight-2 correlated Pauli
    fault after each `Gcnot`. Both patches use `FT Zero Prep` and `FT
    Logical Z Measure`, with one `Adaptive QEC` round per patch after the
    CX -- confirming the composite CX + single QEC round is fault-tolerant
    to any single fault occurring during the CX itself (not just faults
    injected before it, as in `TestSyndromeCouplingAfterCx`).
    """

    def _base_program_and_cx(self):
        q0 = steane_qubits("_0")
        q1 = steane_qubits("_1")
        all_q = q0 + q1
        geometry = PatchGeometry(
            patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout="7_1_3"
        )
        cx_circuit = multipatch.build_transversal_cx_circuit_instruction(
            geometry
        )
        cx_bookkeeping = multipatch.build_cx_bookkeeping_instruction(
            "L0", "L1"
        )
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("Steane"),
            ("FT Zero Prep", "L0"),
            ("FT Zero Prep", "L1"),
            (cx_circuit, None),
            (cx_bookkeeping, None),
            ("Adaptive QEC", "L0"),
            ("Adaptive QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical Z Measure", "L1"),
        ]
        cx_stack_idx = stack.index((cx_circuit, None))
        program = make_program(stack, all_q)
        return program, cx_circuit, cx_stack_idx

    def test_weight1_pre_gate_faults(self):
        """Every single-qubit Pauli fault before each of the CX's 7 Gcnots."""
        program, cx_circuit, cx_idx = self._base_program_and_cx()
        injected = fttools.build_discrete_error_injection_programs(
            base_program=program,
            instruction_to_analyze=cx_circuit,
            stack_idx_to_modify=cx_idx,
            error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
        )
        failed = fttools.run_discrete_error_injected_programs(
            injected,
            [("logical_measurement", "all", True)],
            [[0, 0]],
        )
        assert len(failed) == 0, (
            f"{len(failed)} pre-gate fault(s) not corrected"
        )

    def test_weight2_post_gate_correlated_faults(self):
        """Every correlated weight-2 fault after each of the CX's 7 Gcnots."""
        program, cx_circuit, cx_idx = self._base_program_and_cx()
        injected = fttools.build_discrete_error_injection_programs(
            base_program=program,
            instruction_to_analyze=cx_circuit,
            stack_idx_to_modify=cx_idx,
            error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
            post_twoq_gates=True,
        )
        failed = fttools.run_discrete_error_injected_programs(
            injected,
            [("logical_measurement", "all", True)],
            [[0, 0]],
        )
        assert len(failed) == 0, (
            f"{len(failed)} post-gate fault(s) not corrected"
        )
