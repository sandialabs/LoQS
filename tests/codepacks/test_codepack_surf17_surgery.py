"""Tester for lattice-surgery machinery built on codepack_surf17_tomita2014.

Phase E: seam geometry constants and merged check matrices (pure, no simulator).
Phase F: simplified merge/split surgery (seam-check joint parity, per-patch decoding).
Phase G: FT merged-window decoding.
Phase H: surgery CNOT, tested through its components on two-patch geometries.

These are plumbing tests: each instruction builds, composes into a program
and runs, with the noiseless outcome asserted where that is free.
"""

import itertools

import numpy as np
import pytest

pygsti = pytest.importorskip("pygsti")

from loqs.backends import (
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMQuantumState,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.core import InstructionStack, PatchGeometry, QuantumProgram
from loqs.core.frame import Frame
from loqs.core.instructions import Instruction, builders
from loqs.codepacks import codepack_surf17_surgery as surgery
from loqs.codepacks import codepack_surf17_multipatch as multipatch
from loqs.codepacks import codepack_surf17_tomita2014 as codepack_surf17
from loqs.tools import fttools

NUM_STIM_SHOTS = 10
# The dense-statevector smoke test below is fully deterministic (Zero Prep
# product state, noiseless model), so a single shot is sufficient; the
# dense backend is expensive enough (23-qubit statevector) that repeated
# shots would only add cost without adding coverage.
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


def gf2_rank(M: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2) via Gaussian elimination."""
    M = (np.array(M, dtype=int) % 2).copy()
    rank = 0
    num_rows, num_cols = M.shape
    for col in range(num_cols):
        pivot = None
        for r in range(rank, num_rows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(num_rows):
            if r != rank and M[r, col] == 1:
                M[r] ^= M[rank]
        rank += 1
        if rank == num_rows:
            break
    return rank


def support_vector(support) -> np.ndarray:
    vec = np.zeros(surgery.MERGED_NUM_DATA, dtype=int)
    for elem in support:
        vec[surgery.merged_index(elem)] = 1
    return vec


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


SEAMS = ["Qs0", "Qs1", "Qs2"]


def two_patch_setup(layout):
    """(PatchGeometry roles "a"/"b" = L0/L1, all_qubits incl. seams)."""
    q0 = layout_qubits(layout, "_0")
    q1 = layout_qubits(layout, "_1")
    geometry = PatchGeometry(
        patches={"a": ("L0", q0), "b": ("L1", q1)},
        seam=SEAMS,
        layout=layout,
    )
    return geometry, q0 + q1 + SEAMS


def collect_parities(results, key):
    """Per-shot list of surgery parity values recorded under `key`."""
    return results.collect_shot_data(key, "all", strip_none_entries=True)


class TestSyndromeRowOrdering:
    """Foundation regression: history rows must align with H-matrix rows.

    The surgery bookkeeping substitutes grown-check values into specific
    history row indices, so a mismatch between measurement order and the
    codepack's SyndromeLabel row assignment (as existed for surf10's X
    checks) would silently corrupt the merge-window histories.
    """

    # Single-qubit errors whose syndromes are one-hot in a single H row
    X_PROBES = [("D0", 0), ("D2", 1), ("D8", 2), ("D6", 3)]  # Z errors
    Z_PROBES = [("D0", 0), ("D2", 1), ("D6", 2), ("D8", 3)]  # X errors

    @pytest.mark.parametrize("layout", ["surf17", "surf13", "surf10"])
    @pytest.mark.parametrize("check_type", ["X", "Z"])
    def test_history_rows_match_H_rows(self, layout, check_type):
        qubits = layout_qubits(layout)
        code = codepack_surf17.create_qec_code(layout=layout, num_qec_rounds=1)
        model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=StimCircuitGateRep,
            model_backend=DictNoiseModel,
        )
        if check_type == "X":
            probes, err_gate, prep = self.X_PROBES, "Gzpi", "Plus Prep"
        else:
            probes, err_gate, prep = self.Z_PROBES, "Gxpi", "Zero Prep"

        for data_qubit, expected_row in probes:
            err_circ = PyGSTiPhysicalCircuit(
                [[(err_gate, data_qubit)]], qubit_labels=qubits
            )
            err_inst = builders.build_physical_circuit_instruction(
                err_circ, name="probe error"
            )
            stack = InstructionStack(
                [
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
                    (err_inst, "L0"),
                    ("Syndrome Extraction", "L0"),
                    ("Decoder", "L0"),
                ]
            )
            program = QuantumProgram(
                stack,
                default_noise_model=model,
                state_type=STIMQuantumState,
                patch_types={"SURF": code},
            )
            res = program.run(num_shots=1, verbose=False)
            patch = res.collect_shot_data("patches", -1)[0]["L0"]
            hist = patch.data[f"syndrome_history_{check_type}"]
            expected = [0, 0, 0, 0]
            expected[expected_row] = 1
            assert hist == [expected], (
                f"{layout} {check_type} row ordering: {err_gate} on "
                f"{data_qubit} gave {hist}, expected {[expected]}"
            )


class TestMergedGeometry:
    """Phase E: pure geometry checks of the merged-code matrices."""

    def test_base_matrices_match_codepack(self):
        """Module-level base H matrices mirror the codepack's decoder data."""
        code = codepack_surf17.create_qec_code(layout="surf17")
        inst = code.instructions["FT Z logical parity calculation"]
        assert np.array_equal(surgery.BASE_H_X, inst.data["H_X"])
        assert np.array_equal(surgery.BASE_H_Z, inst.data["H_Z"])

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_merged_checks_commute(self, kind):
        """Every X check overlaps every Z check on an even number of qubits."""
        H_X, H_Z, _, _ = surgery.build_merged_check_matrices(kind)
        assert np.all((H_X @ H_Z.T) % 2 == 0)

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_merged_code_has_one_logical(self, kind):
        """21 data qubits - 20 independent checks = 1 logical qubit."""
        H_X, H_Z, _, _ = surgery.build_merged_check_matrices(kind)
        rank_sum = gf2_rank(H_X) + gf2_rank(H_Z)
        assert rank_sum == H_X.shape[0] + H_Z.shape[0] == 20
        assert surgery.MERGED_NUM_DATA - rank_sum == 1

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_new_checks_telescope_to_joint_logical(self, kind):
        """The product of the 4 seam checks is the joint logical parity.

        ZZ: Z(A.D6,A.D7,A.D8) (x) Z(B.D0,B.D1,B.D2) = Z_L(A) Z_L(B)
        XX: X(A.D2,A.D5,A.D8) (x) X(B.D0,B.D3,B.D6) = X_L(A) X_L(B)
        """
        seam_geometry = surgery.SEAM_GEOMETRIES[kind]
        product = np.zeros(surgery.MERGED_NUM_DATA, dtype=int)
        for check in seam_geometry["new_checks"]:
            product ^= support_vector(check["support"])
        assert np.array_equal(
            product, support_vector(seam_geometry["parity_support"])
        )
        # No seam qubits survive the telescoping
        assert np.all(product[9:12] == 0)

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_parity_support_is_boundary_logical_pair(self, kind):
        """The telescoped support is a logical representative on each patch.

        Each 3-qubit half must differ from the codepack's canonical logical
        (Z0 Z4 Z8 or X2 X4 X6) by a product of that patch's own stabilizers.
        """
        seam_geometry = surgery.SEAM_GEOMETRIES[kind]
        H = surgery.BASE_H_Z if kind == "ZZ" else surgery.BASE_H_X
        canonical = np.zeros(9, dtype=int)
        for i in [0, 4, 8] if kind == "ZZ" else [2, 4, 6]:
            canonical[i] = 1

        for patch in ["A", "B"]:
            half = np.zeros(9, dtype=int)
            for p, i in seam_geometry["parity_support"]:
                if p == patch:
                    half[i] = 1
            diff = half ^ canonical
            # diff must lie in the rowspace of the patch's stabilizer matrix
            stacked = np.vstack([H, diff])
            assert gf2_rank(stacked) == gf2_rank(H), (
                f"{kind} parity support on patch {patch} is not equivalent "
                "to the canonical logical"
            )

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_grown_checks_consistent(self, kind):
        """Grown check = old boundary check (+) its two seam qubits."""
        seam_geometry = surgery.SEAM_GEOMETRIES[kind]
        H_old = (
            surgery.BASE_H_X if kind == "ZZ" else surgery.BASE_H_Z
        )  # grown checks are X-type for ZZ merges, Z-type for XX
        for patch in ["A", "B"]:
            grown = seam_geometry["grown_checks"][patch]
            old_row = H_old[grown["check_row"]]
            old_from_geometry = np.zeros(9, dtype=int)
            for p, i in grown["old_support"]:
                assert p == patch
                old_from_geometry[i] = 1
            assert np.array_equal(old_row, old_from_geometry)
            expected = support_vector(
                grown["old_support"] + grown["seam_pair"]
            )
            assert np.array_equal(expected, support_vector(grown["support"]))

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_merged_logicals(self, kind):
        """Merged Z_L/X_L commute with all checks and anticommute mutually."""
        seam_geometry = surgery.SEAM_GEOMETRIES[kind]
        H_X, H_Z, _, _ = surgery.build_merged_check_matrices(kind)
        z_logical = support_vector(seam_geometry["merged_Z_L"])
        x_logical = support_vector(seam_geometry["merged_X_L"])
        # Z-type logical must commute with all X checks and vice versa
        assert np.all((H_X @ z_logical) % 2 == 0)
        assert np.all((H_Z @ x_logical) % 2 == 0)
        # The pair must anticommute (odd overlap)
        assert (z_logical @ x_logical) % 2 == 1

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_patch_interior_checks_embedded(self, kind):
        """Non-grown patch checks appear verbatim at their patch offsets."""
        seam_geometry = surgery.SEAM_GEOMETRIES[kind]
        grown_type = seam_geometry["grown_check_type"]
        for check_type, H_base in (
            ("X", surgery.BASE_H_X),
            ("Z", surgery.BASE_H_Z),
        ):
            H_X, H_Z, labels_X, labels_Z = surgery.build_merged_check_matrices(
                kind
            )
            H_merged = H_X if check_type == "X" else H_Z
            labels = labels_X if check_type == "X" else labels_Z
            for patch, offset in (("A", 0), ("B", 12)):
                grown_row = (
                    seam_geometry["grown_checks"][patch]["check_row"]
                    if check_type == grown_type
                    else None
                )
                for i in range(4):
                    if i == grown_row:
                        continue
                    label = f"{patch}.S{check_type}{i}"
                    row = H_merged[labels.index(label)]
                    assert np.array_equal(row[offset : offset + 9], H_base[i])
                    # ... and nothing outside the patch block
                    mask = np.ones(21, dtype=bool)
                    mask[offset : offset + 9] = False
                    assert np.all(row[mask] == 0)

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_byproduct_condition_derivation(self, kind):
        """The byproduct condition is the through-seam bit plus grown pickups.

        The through-seam logical crosses exactly one seam qubit (S0). The
        canonical readout representatives additionally pick up the value
        change of every grown check used in the representative conversion,
        each contributing its seam pair. Over GF(2) the total must equal
        the recorded byproduct_seam_indices.
        """
        seam_geometry = surgery.SEAM_GEOMETRIES[kind]
        through_logical = (
            seam_geometry["merged_X_L"]
            if kind == "ZZ"
            else seam_geometry["merged_Z_L"]
        )
        seam_crossings = [e for e in through_logical if e[0] == "S"]
        assert seam_crossings == [("S", 0)]

        # Canonical conversion: for ZZ, X2X4X6(B) = leftcol(B) * SX0 * SX1
        # where SX1 is B's grown check; for XX, Z0Z4Z8(A) = toprow(A) *
        # SZ1 * SZ3 where SZ3 is A's grown check.
        grown_in_conversion = "B" if kind == "ZZ" else "A"
        pickup = set()
        for _, si in seam_geometry["grown_checks"][grown_in_conversion][
            "seam_pair"
        ]:
            pickup ^= {si}
        expected = {0} ^ pickup
        assert set(seam_geometry["byproduct_seam_indices"]) == expected

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_telescope_reference_conversion(self, kind):
        """Declared reference checks convert telescoped to canonical reps.

        For each patch, (telescoped half) XOR (canonical logical) must equal
        the GF(2) sum of the declared telescope_reference_checks rows of the
        seam-check-type H matrix.
        """
        seam_geometry = surgery.SEAM_GEOMETRIES[kind]
        H = surgery.BASE_H_Z if kind == "ZZ" else surgery.BASE_H_X
        canonical = np.zeros(9, dtype=int)
        for i in [0, 4, 8] if kind == "ZZ" else [2, 4, 6]:
            canonical[i] = 1
        for patch in ["A", "B"]:
            half = np.zeros(9, dtype=int)
            for p, i in seam_geometry["parity_support"]:
                if p == patch:
                    half[i] = 1
            ref_sum = np.zeros(9, dtype=int)
            for row in seam_geometry["telescope_reference_checks"][patch]:
                ref_sum ^= H[row]
            assert np.array_equal(half ^ canonical, ref_sum), (
                f"{kind} patch {patch}: telescope_reference_checks do not "
                "convert the telescoped half to the canonical logical"
            )


class TestSimplifiedSurgeryZZ:
    """Phase F: simplified (per-patch-decoded) ZZ lattice surgery."""

    @pytest.mark.parametrize("layout", ["surf10"])
    @pytest.mark.parametrize("logical_x_on_l0", [True])
    def test_product_state_truth_table(self, layout, logical_x_on_l0):
        """|00> gives m_ZZ = 0; X_L on one patch gives 1, every shot.

        The post-surgery destructive FT Z measures also validate that the
        merge window leaves the per-patch Z decoding intact.
        """
        geometry, all_q = two_patch_setup(layout)
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", geometry, mode="simple"
        )
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("SURF"),
            ("Zero Prep", "L0"),
            ("Zero Prep", "L1"),
            *((("X", "L0"),) if logical_x_on_l0 else ()),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (zz, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical Z Measure", "L1"),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        expected = 1 if logical_x_on_l0 else 0
        parities = collect_parities(results, "surgery_parity_zz_L0_L1")
        assert parities == [[expected]] * NUM_STIM_SHOTS
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS


class TestSimplifiedSurgeryXX:
    """Phase F: simplified XX lattice surgery (dual of the ZZ tests)."""

    @pytest.mark.parametrize("layout", ["surf10"])
    @pytest.mark.parametrize("logical_z_on_l0", [True])
    def test_product_state_truth_table(self, layout, logical_z_on_l0):
        """|++> gives m_XX = 0; Z_L on one patch gives 1, every shot."""
        geometry, all_q = two_patch_setup(layout)
        xx = surgery.build_surgery_parity_instruction(
            "XX", geometry, mode="simple"
        )
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("SURF"),
            ("Plus Prep", "L0"),
            ("Plus Prep", "L1"),
            *((("Z", "L0"),) if logical_z_on_l0 else ()),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (xx, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical X Measure", "L0"),
            ("FT Logical X Measure", "L1"),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        expected = 1 if logical_z_on_l0 else 0
        parities = collect_parities(results, "surgery_parity_xx_L0_L1")
        assert parities == [[expected]] * NUM_STIM_SHOTS
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS


class TestSimplifiedSurgeryBell:
    """Phase F: surgery ZZ and XX parities on a Bell pair in one shot."""

    @pytest.mark.parametrize("layout", ["surf13"])
    def test_bell_zz_then_xx(self, layout):
        """Bell pair: sequential surgery M_ZZ and M_XX both report 0.

        Uses distinct seam triples for the vertical (ZZ) and horizontal
        (XX) merges. This is the strongest simplified-mode test: it needs
        the telescope-reference conversion (the CNOT randomizes both
        patches' boundary-stabilizer values), the byproduct injection from
        the first surgery to be consumed correctly by the second, and both
        grown-check rewrites.
        """
        q0 = layout_qubits(layout, "_0")
        q1 = layout_qubits(layout, "_1")
        seams_v = ["Qsv0", "Qsv1", "Qsv2"]
        seams_h = ["Qsh0", "Qsh1", "Qsh2"]
        all_q = q0 + q1 + seams_v + seams_h
        cnot_geometry = PatchGeometry(
            patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout=layout
        )
        zz_geometry = PatchGeometry(
            patches={"a": ("L0", q0), "b": ("L1", q1)},
            seam=seams_v,
            layout=layout,
        )
        xx_geometry = PatchGeometry(
            patches={"a": ("L0", q0), "b": ("L1", q1)},
            seam=seams_h,
            layout=layout,
        )
        cnot = multipatch.build_transversal_cnot_instruction(cnot_geometry)
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", zz_geometry, mode="simple"
        )
        xx = surgery.build_surgery_parity_instruction(
            "XX", xx_geometry, mode="simple"
        )
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
            ("QEC", "L0"),
            ("QEC", "L1"),
            (xx, None),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        assert collect_parities(results, "surgery_parity_zz_L0_L1") == (
            [[0]] * NUM_STIM_SHOTS
        )
        assert collect_parities(results, "surgery_parity_xx_L0_L1") == (
            [[0]] * NUM_STIM_SHOTS
        )


class TestFTSurgery:
    """Phase G: FT surgery through the merged-matching-graph window decode."""

    @pytest.mark.parametrize("layout", ["surf10"])
    @pytest.mark.parametrize("logical_x_on_l0", [True])
    def test_zz_truth_table_ft(self, layout, logical_x_on_l0):
        """FT-mode ZZ truth table: |00> -> 0, X_L(L0) -> 1, every shot."""
        geometry, all_q = two_patch_setup(layout)
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", geometry, mode="ft"
        )
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("SURF"),
            ("Zero Prep", "L0"),
            ("Zero Prep", "L1"),
            *((("X", "L0"),) if logical_x_on_l0 else ()),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (zz, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            ("FT Logical Z Measure", "L1"),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        expected = 1 if logical_x_on_l0 else 0
        assert collect_parities(results, "surgery_parity_zz_L0_L1") == (
            [[expected]] * NUM_STIM_SHOTS
        )
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", ["surf10"])
    @pytest.mark.parametrize("logical_z_on_l0", [True])
    def test_xx_truth_table_ft(self, layout, logical_z_on_l0):
        """FT-mode XX truth table: |++> -> 0, Z_L(L0) -> 1, every shot."""
        geometry, all_q = two_patch_setup(layout)
        xx = surgery.build_surgery_parity_instruction(
            "XX", geometry, mode="ft"
        )
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("SURF"),
            ("Plus Prep", "L0"),
            ("Plus Prep", "L1"),
            *((("Z", "L0"),) if logical_z_on_l0 else ()),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (xx, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical X Measure", "L0"),
            ("FT Logical X Measure", "L1"),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        expected = 1 if logical_z_on_l0 else 0
        assert collect_parities(results, "surgery_parity_xx_L0_L1") == (
            [[expected]] * NUM_STIM_SHOTS
        )
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", ["surf10"])
    def test_bell_zz_then_xx_ft(self, layout):
        """Bell pair: FT-mode surgery M_ZZ and M_XX both report 0."""
        q0 = layout_qubits(layout, "_0")
        q1 = layout_qubits(layout, "_1")
        seams_v = ["Qsv0", "Qsv1", "Qsv2"]
        seams_h = ["Qsh0", "Qsh1", "Qsh2"]
        all_q = q0 + q1 + seams_v + seams_h
        cnot_geometry = PatchGeometry(
            patches={"ctrl": ("L0", q0), "tgt": ("L1", q1)}, layout=layout
        )
        zz_geometry = PatchGeometry(
            patches={"a": ("L0", q0), "b": ("L1", q1)},
            seam=seams_v,
            layout=layout,
        )
        xx_geometry = PatchGeometry(
            patches={"a": ("L0", q0), "b": ("L1", q1)},
            seam=seams_h,
            layout=layout,
        )
        cnot = multipatch.build_transversal_cnot_instruction(cnot_geometry)
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", zz_geometry, mode="ft"
        )
        xx = surgery.build_surgery_parity_instruction(
            "XX", xx_geometry, mode="ft"
        )
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
            ("QEC", "L0"),
            ("QEC", "L1"),
            (xx, None),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        assert collect_parities(results, "surgery_parity_zz_L0_L1") == (
            [[0]] * NUM_STIM_SHOTS
        )
        assert collect_parities(results, "surgery_parity_xx_L0_L1") == (
            [[0]] * NUM_STIM_SHOTS
        )


class TestParityReadoutConsistencyA:
    """Regression: the parity/readout identity with patch A in a random basis.

    Phase F's FT-consistency tests only exercised the B side (e.g. |0+> for
    ZZ), where patch A's relevant syndrome rows are deterministic. With A
    prepped in the conjugate basis of the parity, A's round-0 syndrome layer
    is random, and the destructive FT measure of A needs the prep-conjugate
    reference-round flag — without it the decoder matches round-0 values as
    real defects and applies a random logical correction, which masquerades
    as a surgery bookkeeping bug (m looks inconsistent with the readouts).
    """

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    @pytest.mark.parametrize("mode", ["simple", "ft"])
    def test_parity_matches_ft_readouts_random_patch_a(self, kind, mode):
        """A random-basis: readout_A ^ readout_B == m every shot."""
        layout = "surf10"
        geometry, all_q = two_patch_setup(layout)
        inst = surgery.build_surgery_parity_instruction(
            kind, geometry, mode=mode
        )
        if kind == "ZZ":
            prep0, prep1 = "Plus Prep", "Zero Prep"
            meas = "FT Logical Z Measure"
            flag = {"reference_round_mode_Z": "guarded_diff"}
        else:
            prep0, prep1 = "Zero Prep", "Plus Prep"
            meas = "FT Logical X Measure"
            flag = {"reference_round_mode_X": "guarded_diff"}
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("SURF"),
            (prep0, "L0"),
            (prep1, "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (inst, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            {"instruction": meas, "patch_label": "L0", **flag},
            (meas, "L1"),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        parities = [
            p[0]
            for p in collect_parities(
                results, f"surgery_parity_{kind.lower()}_L0_L1"
            )
        ]
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(
            (logicals[i][0] ^ logicals[i][1]) == parities[i]
            for i in range(NUM_STIM_SHOTS)
        )


class TestSurgeryDenseSmoke:
    """Phase F: statevector smoke on the intended dense configuration."""

    @pytest.mark.slow
    def test_zz_statevector_surf10(self):
        """2x surf10 + 3 seams (23 qubits): |00> -> m_ZZ = 0."""
        layout = "surf10"
        geometry, all_q = two_patch_setup(layout)
        # 2 merge rounds keep the 23-qubit statevector runtime tolerable;
        # noiseless round parities are constant so the vote is unaffected.
        zz = surgery.build_surgery_parity_instruction(
            "ZZ",
            geometry,
            mode="simple",
            num_merge_rounds=2,
        )
        stack = InstructionStack(
            [
                {
                    "instruction": "Init State",
                    "state": len(all_q),
                    "qubit_labels": all_q,
                },
                *geometry.init_patch_entries("SURF"),
                ("Zero Prep", "L0"),
                ("Zero Prep", "L1"),
                ("QEC", "L0"),
                ("QEC", "L1"),
                (zz, None),
            ]
        )
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
        parities = collect_parities(results, "surgery_parity_zz_L0_L1")
        assert parities == [[0]] * NUM_KRAUS_SHOTS


class TestMzzBellPrep:
    """Phase I: Bell prep by a direct M_ZZ merge on |+>_L |+>_L.

    build_mzz_bell_prep_sequence measures Z_L Z_L between the two patches
    (no ancilla patch) and injects the conditional frame correction
    X_L(L1)^m_zz, leaving (|00> + |11>)/sqrt(2). Verified by perfect
    XOR-parity in destructive FT readouts of BOTH bases.
    """

    @staticmethod
    def bell_prep_stack(layout, geometry, all_q, mode, basis):
        """|+>|+> -> M_ZZ Bell prep -> QEC -> FT readout of both patches."""
        seq = surgery.build_mzz_bell_prep_sequence(geometry, mode=mode)
        stack = [
            {
                "instruction": "Init State",
                "state": len(all_q),
                "qubit_labels": all_q,
            },
            *geometry.init_patch_entries("SURF"),
            ("Plus Prep", "L0"),
            ("Plus Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            *seq,
            ("QEC", "L0"),
            ("QEC", "L1"),
        ]
        # |+> prep -> random round-0 Z layer ("guarded_diff"); the ZZ
        # merge grows an X check on both patches (also "guarded_diff").
        flag = {f"reference_round_mode_{basis}": "guarded_diff"}
        stack += [
            {
                "instruction": f"FT Logical {basis} Measure",
                "patch_label": "L0",
                **flag,
            },
            {
                "instruction": f"FT Logical {basis} Measure",
                "patch_label": "L1",
                **flag,
            },
        ]
        return stack

    @pytest.mark.parametrize("mode", ["simple", "ft"])
    def test_bell_prep_correlations_and_correction(self, mode):
        """Z and X readouts XOR to 0 every shot; the correction equals m_zz.

        Without the conditional X_L correction the Z-basis XOR would equal
        m_zz, so a zero XOR shows the correction fires on exactly the
        m_zz = 1 shots. The X frame is invisible in the X basis.
        """
        layout = "surf10"
        geometry, all_q = two_patch_setup(layout)
        for basis in ("Z", "X"):
            stack = self.bell_prep_stack(layout, geometry, all_q, mode, basis)
            program = make_stim_program(layout, stack, all_q)
            results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
            ms = [
                p[0]
                for p in collect_parities(results, "surgery_parity_zz_L0_L1")
            ]
            logicals = results.collect_shot_data(
                "logical_measurement", "all", strip_none_entries=True
            )
            assert all(shot[0] ^ shot[1] == 0 for shot in logicals), basis
            corrections = results.collect_shot_data(
                "mzz_bell_correction_L0_L1", "all", strip_none_entries=True
            )
            assert [c[0] for c in corrections] == ms, basis


class TestSurgeryCnotComponents:
    """Phase H: the surgery CNOT's components on two-patch geometries.

    The full CNOT needs three patches and two seams (36 qubits even at
    surf10), so its pieces run separately on at most 23 qubits: the
    correction table, the repair's fire path after a ZZ merge, and a
    ZZ-then-XX merge pair on the same two patches.
    """

    LAYOUT = "surf10"
    # Stack entries before the ZZ merge in `_zz_repair_program`: init state,
    # two init patches, prep and QEC on C, prep and QEC on ANC.
    NUM_PRELUDE_ENTRIES = 7
    # Index of the seam-measurement entry in the flat ZZ merge sequence.
    SEAM_MEASURE_IDX = 5

    @staticmethod
    def _init_state(all_q):
        return {
            "instruction": "Init State",
            "state": len(all_q),
            "qubit_labels": all_q,
        }

    @classmethod
    def _zz_repair_geometry(cls):
        qc = layout_qubits(cls.LAYOUT, "_c")
        qa = layout_qubits(cls.LAYOUT, "_a")
        seam = ["Qsv0", "Qsv1", "Qsv2"]
        geometry = PatchGeometry(
            patches={"a": ("C", qc), "b": ("ANC", qa)},
            seam=seam,
            layout=cls.LAYOUT,
        )
        return geometry, qc + qa + seam

    @classmethod
    def _zz_repair_program(cls, c_flip, flat_merge=False, before_measures=()):
        """ZZ merge of C and ANC, then the repair the CNOT uses.

        With `flat_merge`, the merge is the flat list of per-step
        instructions from `build_surgery_parity_instruction_sequence`, so
        one entry can be modified to take a fault. `before_measures` go
        between the repair and the two final measures.
        """
        geometry, all_q = cls._zz_repair_geometry()
        if flat_merge:
            seq = surgery.build_surgery_parity_instruction_sequence(
                "ZZ", geometry, mode="ft"
            )
            zz_entries = [(inst, None) for inst in seq]
        else:
            zz = surgery.build_surgery_parity_instruction(
                "ZZ", geometry, mode="ft"
            )
            zz_entries = [(zz, None)]
        repair = surgery.build_split_byproduct_repair_instruction(
            "C",
            "ANC",
            num_merge_rounds=3,
            num_post_split_rounds=3,
            fire_rule="b_only",
            defect_decode_mode="matching",
        )
        stack = [
            cls._init_state(all_q),
            *geometry.init_patch_entries("SURF"),
            ("Zero Prep", "C"),
            *([("X", "C")] if c_flip else []),
            ("QEC", "C"),
            ("Plus Prep", "ANC"),
            ("QEC", "ANC"),
            *zz_entries,
            ("QEC", "C"),
            ("QEC", "ANC"),
            (repair, None),
            *before_measures,
            {
                "instruction": "FT Logical Z Measure",
                "patch_label": "ANC",
                "reference_round_mode_Z": "guarded_diff",
            },
            {"instruction": "FT Logical Z Measure", "patch_label": "C"},
        ]
        return make_stim_program(cls.LAYOUT, stack, all_q)

    @staticmethod
    def _stub_parities(patches, mzz, mxx, manc):
        return Frame(
            {
                "surgery_parity_zz_C_ANC": mzz,
                "surgery_parity_xx_ANC_T": mxx,
                "logical_measurement": manc,
            }
        )

    @pytest.mark.parametrize(
        "m_zz, m_xx, m_anc", list(itertools.product((0, 1), repeat=3))
    )
    def test_cnot_correction_table(self, m_zz, m_xx, m_anc):
        """Corrections are (m_xx, m_zz ^ m_anc), applied as frame Paulis."""
        qc = layout_qubits(self.LAYOUT, "_c")
        qt = layout_qubits(self.LAYOUT, "_t")
        all_q = qc + qt
        stub = Instruction(
            self._stub_parities,
            data={"mzz": m_zz, "mxx": m_xx, "manc": m_anc},
            name="stub parities",
        )
        corrections = surgery.build_surgery_cnot_corrections_instruction(
            "C", "T", "ANC"
        )
        stack = [
            self._init_state(all_q),
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "C",
                "qubits": qc,
            },
            {
                "instruction": "Init Patch SURF",
                "new_patch_label": "T",
                "qubits": qt,
            },
            (stub, None),
            (corrections, None),
        ]
        program = make_stim_program(self.LAYOUT, stack, all_q)
        results = program.run(num_shots=1, verbose=False)

        recorded = results.collect_shot_data(
            "surgery_cnot_corrections_C_T", -1
        )[0]
        z_ctrl, x_tgt = m_xx, m_zz ^ m_anc
        assert tuple(recorded) == (z_ctrl, x_tgt)

        last = results.collect_shot_data("patches", -1)[0]

        def frame(label):
            pf = last[label].pauli_frame
            return {
                q: p
                for q, p in zip(pf.qubit_labels, pf.pauli_frame)
                if p != "I"
            }

        expected_ctrl = {f"D{i}_c": "Z" for i in (0, 4, 8)} if z_ctrl else {}
        expected_tgt = {f"D{i}_t": "X" for i in (2, 4, 6)} if x_tgt else {}
        assert frame("C") == expected_ctrl
        assert frame("T") == expected_tgt

    def test_repair_fire_path(self):
        """A seam-measurement fault fires the repair; the run completes."""
        program = self._zz_repair_program(0, flat_merge=True)
        idx = self.NUM_PRELUDE_ENTRIES + self.SEAM_MEASURE_IDX
        # seam[2], the ANC-side outer seam qubit, at layer 0.
        injected = fttools.build_discrete_error_injection_program_for_combo(
            program, idx, [(0, "Gzpi", 2)]
        )
        results = injected.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        repairs = collect_parities(results, "split_byproduct_repair_zz_C_ANC")
        assert repairs == [[1]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("c_flip", [0, 1])
    def test_zz_then_xx_merge_pair(self, c_flip):
        """ZZ(C, ANC) with repair, then XX(ANC, C): the CNOT's merge pair.

        C stands in for the target: it gets one extra QEC round before the
        XX merge, so the two patches' histories differ in depth, and ANC's
        history was settled by the ZZ repair.
        """
        geometry_zz, _ = self._zz_repair_geometry()
        geometry_xx = PatchGeometry(
            patches={
                "a": ("ANC", layout_qubits(self.LAYOUT, "_a")),
                "b": ("C", layout_qubits(self.LAYOUT, "_c")),
            },
            seam=geometry_zz.seam,
            layout=self.LAYOUT,
        )
        xx = surgery.build_surgery_parity_instruction(
            "XX", geometry_xx, mode="ft"
        )
        xx_repair = surgery.build_split_byproduct_repair_instruction(
            "ANC", "C"
        )
        program = self._zz_repair_program(
            c_flip,
            before_measures=[
                ("QEC", "C"),
                (xx, None),
                ("QEC", "ANC"),
                ("QEC", "C"),
                (xx_repair, None),
            ],
        )
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        mzz = collect_parities(results, "surgery_parity_zz_C_ANC")
        repairs = collect_parities(results, "split_byproduct_repair_zz_C_ANC")
        # Per-shot entries: [m_anc, m_C].
        for (m_anc, m_c), (z,), (fired,) in zip(logicals, mzz, repairs):
            assert m_c ^ m_anc == z
            assert fired == 0
