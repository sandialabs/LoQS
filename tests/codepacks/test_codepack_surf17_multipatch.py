"""Tester for multi-patch surface-code machinery built on codepack_surf17_tomita2014.

Phase A: per-patch syndrome-history namespacing and reference-round decoding.
Phase B: transversal logical CNOT.
Phase C: ancilla-mediated joint logical parity measurements.
"""

import warnings

import pytest

pygsti = pytest.importorskip("pygsti")

from loqs.backends import (
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMQuantumState,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.core import PatchGeometry, QuantumProgram
from loqs.core.recordables.pauliframe import PauliFrame
from loqs.codepacks import codepack_surf17_tomita2014 as codepack_surf17
from loqs.codepacks import codepack_surf17_multipatch as multipatch
from loqs.tools import fttools

NUM_STIM_SHOTS = 100
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
        stack,
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={"SURF": code},
    )


class TestTwoPatchFoundations:
    """Phase A: per-patch tracked syndrome histories + reference rounds."""

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
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
            {"instruction": "Init State", "state": len(all_q), "qubit_labels": all_q},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": q0},
            {"instruction": "Init Patch SURF", "new_patch_label": "L1", "qubits": q1},
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

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_reference_round_noiseless(self, layout, basis):
        """"clean_diff" mode is still deterministic when noiseless.

        Prep matches the measurement basis here, so round 0 is actually
        deterministic; "clean_diff" mode (drop round 0 as its own detector,
        diff round 1 against it, no escape edges) is the correct choice.
        """
        qubits = layout_qubits(layout)
        prep = "Zero Prep" if basis == "Z" else "Plus Prep"
        meas = f"FT Logical {basis} Measure"
        ref_kwarg = {f"reference_round_mode_{basis}": "clean_diff"}

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": qubits},
            (prep, "L0"),
            ("QEC", "L0"),
            {"instruction": meas, "patch_label": "L0", **ref_kwarg},
        ]

        program = make_stim_program(layout, stack, qubits)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        outcomes = results.collect_shot_data("logical_measurement", -1)
        assert outcomes == [0] * NUM_STIM_SHOTS

    @pytest.mark.slow
    def test_reference_round_ft_semantics(self):
        """Reference round drops the round-0 detector layer -- and only that.

        Zero Prep already makes round 0 deterministic, so this exercises
        "clean_diff" mode: the round-0 layer is dropped and round 1 is
        diffed against it, but round 0 itself isn't assumed noisy (no
        escape edges), unlike the "guarded_diff" mode genuinely-random-prep
        callers need. Single faults injected into syndrome-extraction
        rounds 2 and 3 must still be fully corrected. Single faults in
        round 1 are provably ambiguous under a reference round (the
        round-0 diff detector is gone, so a persistent fault there is
        identical, before and after, to having no fault at all), so at
        least one injection there must produce a logical failure. The
        latter also proves the reference_round_mode_Z kwarg actually
        reaches the decoder through the composite instruction.
        """
        layout = "surf17"
        qubits = layout_qubits(layout)
        code = codepack_surf17.create_qec_code(layout=layout, num_qec_rounds=3)
        model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=StimCircuitGateRep,
            model_backend=DictNoiseModel,
        )

        stack = [
            {"instruction": "Init State", "state": len(qubits), "qubit_labels": qubits},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": qubits},
            ("Zero Prep", "L0"),
            ("Syndrome Extraction", "L0"),  # index 3 (round 1)
            ("Syndrome Extraction", "L0"),  # index 4 (round 2)
            ("Syndrome Extraction", "L0"),  # index 5 (round 3)
            ("Decoder", "L0"),
            {
                "instruction": "FT Logical Z Measure",
                "patch_label": "L0",
                "reference_round_mode_Z": "clean_diff",
            },
        ]

        base_program = QuantumProgram(
            stack,
            default_noise_model=model,
            state_type=STIMQuantumState,
            patch_types={"SURF": code},
            name="Reference-round FT Test",
        )

        # Rounds 2 and 3: all single faults corrected
        for stack_idx in [4, 5]:
            injected = fttools.build_discrete_error_injection_programs(
                base_program=base_program,
                instruction_to_analyze=code.instructions[
                    "Syndrome Extraction"
                ],
                stack_idx_to_modify=stack_idx,
                error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
            )
            failed = fttools.run_discrete_error_injected_programs(
                injected,
                [("logical_measurement", -1)],
                [0],
            )
            assert (
                len(failed) == 0
            ), f"Reference round broke FT for round at stack idx {stack_idx}"

        # Round 1: reference round is blind to some pre-round-0-boundary
        # faults by construction; if the kwarg were silently dropped this
        # would be 0 (the default decode corrects all of these)
        injected = fttools.build_discrete_error_injection_programs(
            base_program=base_program,
            instruction_to_analyze=code.instructions["Syndrome Extraction"],
            stack_idx_to_modify=3,
            error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
        )
        failed = fttools.run_discrete_error_injected_programs(
            injected,
            [("logical_measurement", -1)],
            [0],
        )
        assert len(failed) > 0


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


def two_patch_cnot_stack(layout, prep_ctrl, prep_tgt, meas, meas_kwargs,
                         after_prep=()):
    """Stack: preps, QEC, transversal CNOT L0->L1, QEC, FT measures."""
    q0 = layout_qubits(layout, "_0")
    q1 = layout_qubits(layout, "_1")
    all_q = q0 + q1
    geometry = PatchGeometry(
        patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout=layout
    )
    cnot = multipatch.build_transversal_cnot_instruction(geometry)
    stack = [
        {"instruction": "Init State", "state": len(all_q), "qubit_labels": all_q},
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
    """Phase B: transversal logical CNOT (truth table, Bell correlations, FT)."""

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    def test_zero_zero(self, layout):
        """|0>|0> -> CNOT -> |0>|0>: both Z measures deterministic 0."""
        stack, all_q = two_patch_cnot_stack(
            layout, "Zero Prep", "Zero Prep", "FT Logical Z Measure", {}
        )
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert per_shot == [[0, 0]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
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

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    def test_bell_zz_correlations(self, layout):
        """Bell pair: per-patch Z outcomes are random but always equal.

        Plus Prep on L0 makes the round-0 Z syndromes genuinely random (not
        just non-deterministic-in-expectation) on L0, and the CNOT history
        XOR spreads that to L1, so BOTH measures decode with
        reference_round_mode_Z="guarded_diff".
        """
        stack, all_q = two_patch_cnot_stack(
            layout,
            "Plus Prep",
            "Zero Prep",
            "FT Logical Z Measure",
            {"reference_round_mode_Z": "guarded_diff"},
        )
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert len(per_shot) == NUM_STIM_SHOTS
        for l0, l1 in per_shot:
            assert l0 ^ l1 == 0
        # Both Bell branches show up (P(miss) = 2*2^-100)
        assert {tuple(s) for s in per_shot} == {(0, 0), (1, 1)}

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    def test_bell_xx_correlations(self, layout):
        """Bell pair: per-patch X outcomes are random but always equal.

        Zero Prep on L1 makes the round-0 X syndromes genuinely random (not
        just non-deterministic-in-expectation) on L1, and the CNOT history
        XOR spreads that to L0, so BOTH measures decode with
        reference_round_mode_X="guarded_diff".
        """
        stack, all_q = two_patch_cnot_stack(
            layout,
            "Plus Prep",
            "Zero Prep",
            "FT Logical X Measure",
            {"reference_round_mode_X": "guarded_diff"},
        )
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert len(per_shot) == NUM_STIM_SHOTS
        for l0, l1 in per_shot:
            assert l0 ^ l1 == 0
        assert {tuple(s) for s in per_shot} == {(0, 0), (1, 1)}

    @pytest.mark.slow
    def test_fault_tolerance(self):
        """Single faults around the CNOT never cause a logical error.

        Zero/Zero prep + Z measures (no reference rounds), so per-patch
        outcomes are individually fault-tolerant observables. Injection
        sites cover every error-copy direction of the history XOR rule:

        - pre-CNOT SE on L0 (X errors copy ctrl -> tgt at the CNOT)
        - pre-CNOT SE on L1 (Z errors copy tgt -> ctrl at the CNOT)
        - weight-1 faults inside the CNOT circuit
        - weight-2 correlated faults after each physical CNOT
        - post-CNOT SE on L0 (plain decoding after the XOR)
        """
        layout = "surf17"
        q0 = layout_qubits(layout, "_0")
        q1 = layout_qubits(layout, "_1")
        all_q = q0 + q1
        code = codepack_surf17.create_qec_code(layout=layout, num_qec_rounds=3)
        model = codepack_surf17.create_ideal_model(
            all_q,
            gaterep=StimCircuitGateRep,
            model_backend=DictNoiseModel,
        )

        cnot_circ = multipatch.build_transversal_cnot_circuit_instruction(
            q0[:9], q1[:9]
        )
        cnot_book = multipatch.build_cnot_bookkeeping_instruction("L0", "L1")

        stack = [
            {"instruction": "Init State", "state": len(all_q), "qubit_labels": all_q},
            {"instruction": "Init Patch SURF", "new_patch_label": "L0", "qubits": q0},
            {"instruction": "Init Patch SURF", "new_patch_label": "L1", "qubits": q1},
            ("Zero Prep", "L0"),
            ("Zero Prep", "L1"),
            ("Syndrome Extraction", "L0"),  # 5 <- inject
            ("Syndrome Extraction", "L0"),  # 6
            ("Syndrome Extraction", "L0"),  # 7
            ("Decoder", "L0"),  # 8
            ("Syndrome Extraction", "L1"),  # 9 <- inject
            ("Syndrome Extraction", "L1"),  # 10
            ("Syndrome Extraction", "L1"),  # 11
            ("Decoder", "L1"),  # 12
            (cnot_circ, None),  # 13 <- inject (weight-1 and weight-2)
            (cnot_book, None),  # 14
            ("Syndrome Extraction", "L0"),  # 15 <- inject
            ("Syndrome Extraction", "L0"),  # 16
            ("Syndrome Extraction", "L0"),  # 17
            ("Decoder", "L0"),  # 18
            ("Syndrome Extraction", "L1"),  # 19
            ("Syndrome Extraction", "L1"),  # 20
            ("Syndrome Extraction", "L1"),  # 21
            ("Decoder", "L1"),  # 22
            ("FT Logical Z Measure", "L0"),  # 23
            ("FT Logical Z Measure", "L1"),  # 24
        ]

        base_program = QuantumProgram(
            stack,
            default_noise_model=model,
            state_type=STIMQuantumState,
            patch_types={"SURF": code},
            name="Transversal CNOT FT Test",
        )

        injection_sets = [
            (
                "SE L0 pre-CNOT",
                code.instructions["Syndrome Extraction"],
                5,
                False,
            ),
            (
                "SE L1 pre-CNOT",
                code.instructions["Syndrome Extraction"],
                9,
                False,
            ),
            ("CNOT weight-1", cnot_circ, 13, False),
            ("CNOT weight-2 post", cnot_circ, 13, True),
            (
                "SE L0 post-CNOT",
                code.instructions["Syndrome Extraction"],
                15,
                False,
            ),
        ]

        for tag, inst, stack_idx, post_twoq in injection_sets:
            injected = fttools.build_discrete_error_injection_programs(
                base_program=base_program,
                instruction_to_analyze=inst,
                stack_idx_to_modify=stack_idx,
                error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
                post_twoq_gates=post_twoq,
            )
            failed = fttools.run_discrete_error_injected_programs(
                injected,
                [("logical_measurement", "all", True)],
                [[0, 0]],
            )
            assert len(failed) == 0, (
                f"{tag}: {len(failed)}/{len(injected)} injected programs "
                "caused a logical error on some patch"
            )


def bell_joint_parity_stack(layout, ancilla="Qanc", ft_measures=False):
    """Bell prep (Plus L0, Zero L1, transversal CNOT) + joint ZZ + XX parity.

    If `ft_measures` is set, both parities are followed by destructive FT
    Z measures on both patches (with reference rounds, as required for
    Bell-type preps), which checks the parity measurements are
    non-destructive.
    """
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
        {"instruction": "Init State", "state": len(all_q), "qubit_labels": all_q},
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
    if ft_measures:
        stack += [
            {
                "instruction": "FT Logical Z Measure",
                "patch_label": "L0",
                "reference_round_mode_Z": "guarded_diff",
            },
            {
                "instruction": "FT Logical Z Measure",
                "patch_label": "L1",
                "reference_round_mode_Z": "guarded_diff",
            },
        ]
    return stack, all_q


class TestJointParity:
    """Phase C: ancilla-mediated joint ZZ / XX logical parity measurements."""

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    @pytest.mark.parametrize("logical_x_on_l0", [False, True])
    def test_joint_zz_product_states(self, layout, logical_x_on_l0):
        """|0>|0> has even joint Z parity; X_L|0>|0> has odd parity."""
        q0 = layout_qubits(layout, "_0")
        q1 = layout_qubits(layout, "_1")
        ancilla = "Qanc"
        all_q = q0 + q1 + [ancilla]
        geometry = PatchGeometry(
            patches={"a": ("L0", q0), "b": ("L1", q1)}, layout=layout
        )
        zz = multipatch.build_joint_parity_zz_instruction(geometry, ancilla)
        stack = [
            {"instruction": "Init State", "state": len(all_q), "qubit_labels": all_q},
            *geometry.init_patch_entries("SURF"),
            ("Zero Prep", "L0"),
            ("Zero Prep", "L1"),
            *((("X", "L0"),) if logical_x_on_l0 else ()),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (zz, None),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        parities = results.collect_shot_data(
            "joint_parity_zz_L0_L1", "all", strip_none_entries=True
        )
        expected = 1 if logical_x_on_l0 else 0
        assert parities == [[expected]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
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

    def test_joint_parities_nondestructive(self):
        """After both joint parities, destructive FT Z measures still agree.

        The per-shot XOR of the destructive per-patch Z outcomes must match
        the ancilla-measured ZZ parity (both are the Bell stabilizer, 0).
        """
        layout = "surf17"
        stack, all_q = bell_joint_parity_stack(layout, ft_measures=True)
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        zz = results.collect_shot_data(
            "joint_parity_zz_L0_L1", "all", strip_none_entries=True
        )
        per_shot = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert zz == [[0]] * NUM_STIM_SHOTS
        for l0, l1 in per_shot:
            assert l0 ^ l1 == 0
        # Both Bell branches appear in the destructive readout
        assert {tuple(s) for s in per_shot} == {(0, 0), (1, 1)}


class TestBackendFeasibilityWarning:
    """Phase C: clear warning for dense backends at infeasible qubit counts."""

    def test_infeasible_hard_warning(self, capsys):
        with pytest.warns(UserWarning, match="INFEASIBLE"):
            n = multipatch.warn_if_backend_infeasible("surf17", "kraus")
        assert n == 34
        assert "INFEASIBLE" in capsys.readouterr().err

    def test_borderline_caution_warning(self, capsys):
        with pytest.warns(UserWarning, match="slow shots"):
            n = multipatch.warn_if_backend_infeasible(
                "surf13", "statevector", extra_qubits=1
            )
        assert n == 27
        assert "WARNING" in capsys.readouterr().err

    def test_stim_always_silent(self, capsys):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            n = multipatch.warn_if_backend_infeasible(
                "surf17", STIMQuantumState
            )
        assert n == 34
        assert capsys.readouterr().err == ""

    def test_small_dense_silent(self, capsys):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            n = multipatch.warn_if_backend_infeasible(
                "surf10", "kraus", extra_qubits=1
            )
        assert n == 21
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize(
        "layout, expected_qubits",
        [("surf17", 57), ("surf13", 45), ("surf10", 36)],
    )
    def test_surgery_cnot_dense_infeasible(
        self, layout, expected_qubits, capsys
    ):
        """Cycle 2: 3 patches + 2 seams (surgery CNOT) is dense-INFEASIBLE."""
        with pytest.warns(UserWarning, match="INFEASIBLE"):
            n = multipatch.warn_if_backend_infeasible(
                layout, "kraus", n_patches=3, extra_qubits=6
            )
        assert n == expected_qubits
        assert "INFEASIBLE" in capsys.readouterr().err

    def test_surgery_cnot_stim_silent(self, capsys):
        """The surgery-CNOT register never warns on the stim backend."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            n = multipatch.warn_if_backend_infeasible(
                "surf17", STIMQuantumState, n_patches=3, extra_qubits=6
            )
        assert n == 57
        assert capsys.readouterr().err == ""

    def test_surgery_joint_surf10_kraus_caution(self, capsys):
        """2x surf10 + 2 seams (26 qubits) gets the gentle kraus warning."""
        with pytest.warns(UserWarning, match="slow shots"):
            n = multipatch.warn_if_backend_infeasible(
                "surf10", "kraus", n_patches=2, extra_qubits=6
            )
        assert n == 26
        assert "WARNING" in capsys.readouterr().err


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
