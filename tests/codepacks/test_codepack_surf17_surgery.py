"""Tester for lattice-surgery machinery built on codepack_surf17_tomita2014.

Phase E: seam geometry constants and merged check matrices (pure, no simulator).
Phase F: simplified merge/split surgery (seam-check joint parity, per-patch decoding).
Phase G: FT merged-window decoding.
Phase H: surgery CNOT.
"""

import numpy as np
import pytest

pygsti = pytest.importorskip("pygsti")

from loqs.backends import (
    GateRep,
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMQuantumState,
)
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.core import QuantumProgram
from loqs.core.instructions import builders
from loqs.codepacks import codepack_surf17_surgery as surgery
from loqs.codepacks import codepack_surf17_multipatch as multipatch
from loqs.codepacks import codepack_surf17_tomita2014 as codepack_surf17
from loqs.tools import fttools

NUM_STIM_SHOTS = 100
NUM_KRAUS_SHOTS = 4


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
        gaterep=GateRep.STIM_CIRCUIT_STR,
        model_backend=DictNoiseModel,
    )
    return QuantumProgram(
        stack,
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={"SURF": code},
    )


SEAMS = ["S0", "S1", "S2"]


def two_patch_setup(layout):
    """(q0, q1, all_qubits incl. seams) for a two-patch surgery program."""
    q0 = layout_qubits(layout, "_0")
    q1 = layout_qubits(layout, "_1")
    return q0, q1, q0 + q1 + SEAMS


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
        code = codepack_surf17.create_qec_code(
            layout=layout, num_qec_rounds=1
        )
        model = codepack_surf17.create_ideal_model(
            qubits,
            gaterep=GateRep.STIM_CIRCUIT_STR,
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
            stack = [
                ("Init State", None, (len(qubits),), {"qubit_labels": qubits}),
                ("Init Patch SURF", None, ("L0", qubits)),
                (prep, "L0"),
                (err_inst, "L0"),
                ("Syndrome Extraction", "L0"),
                ("Decoder", "L0"),
            ]
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
    def test_merged_shapes_and_labels(self, kind):
        H_X, H_Z, labels_X, labels_Z = surgery.build_merged_check_matrices(
            kind
        )
        if kind == "ZZ":
            assert H_X.shape == (8, 21) and H_Z.shape == (12, 21)
        else:
            assert H_X.shape == (12, 21) and H_Z.shape == (8, 21)
        assert len(labels_X) == H_X.shape[0]
        assert len(labels_Z) == H_Z.shape[0]
        # Exactly two grown checks, on the grown-check type only
        grown = [
            lbl
            for lbl in labels_X + labels_Z
            if lbl.endswith("_grown")
        ]
        assert len(grown) == 2

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
        geometry = surgery.SEAM_GEOMETRIES[kind]
        product = np.zeros(surgery.MERGED_NUM_DATA, dtype=int)
        for check in geometry["new_checks"]:
            product ^= support_vector(check["support"])
        assert np.array_equal(
            product, support_vector(geometry["parity_support"])
        )
        # No seam qubits survive the telescoping
        assert np.all(product[9:12] == 0)

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_parity_support_is_boundary_logical_pair(self, kind):
        """The telescoped support is a logical representative on each patch.

        Each 3-qubit half must differ from the codepack's canonical logical
        (Z0 Z4 Z8 or X2 X4 X6) by a product of that patch's own stabilizers.
        """
        geometry = surgery.SEAM_GEOMETRIES[kind]
        H = surgery.BASE_H_Z if kind == "ZZ" else surgery.BASE_H_X
        canonical = np.zeros(9, dtype=int)
        for i in [0, 4, 8] if kind == "ZZ" else [2, 4, 6]:
            canonical[i] = 1

        for patch in ["A", "B"]:
            half = np.zeros(9, dtype=int)
            for p, i in geometry["parity_support"]:
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
        geometry = surgery.SEAM_GEOMETRIES[kind]
        H_old = (
            surgery.BASE_H_X if kind == "ZZ" else surgery.BASE_H_Z
        )  # grown checks are X-type for ZZ merges, Z-type for XX
        for patch in ["A", "B"]:
            grown = geometry["grown_checks"][patch]
            old_row = H_old[grown["check_row"]]
            old_from_geometry = np.zeros(9, dtype=int)
            for p, i in grown["old_support"]:
                assert p == patch
                old_from_geometry[i] = 1
            assert np.array_equal(old_row, old_from_geometry)
            expected = support_vector(
                grown["old_support"] + grown["seam_pair"]
            )
            assert np.array_equal(
                expected, support_vector(grown["support"])
            )

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_merged_logicals(self, kind):
        """Merged Z_L/X_L commute with all checks and anticommute mutually."""
        geometry = surgery.SEAM_GEOMETRIES[kind]
        H_X, H_Z, _, _ = surgery.build_merged_check_matrices(kind)
        z_logical = support_vector(geometry["merged_Z_L"])
        x_logical = support_vector(geometry["merged_X_L"])
        # Z-type logical must commute with all X checks and vice versa
        assert np.all((H_X @ z_logical) % 2 == 0)
        assert np.all((H_Z @ x_logical) % 2 == 0)
        # The pair must anticommute (odd overlap)
        assert (z_logical @ x_logical) % 2 == 1

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_patch_interior_checks_embedded(self, kind):
        """Non-grown patch checks appear verbatim at their patch offsets."""
        geometry = surgery.SEAM_GEOMETRIES[kind]
        grown_type = geometry["grown_check_type"]
        for check_type, H_base in (("X", surgery.BASE_H_X), ("Z", surgery.BASE_H_Z)):
            H_X, H_Z, labels_X, labels_Z = (
                surgery.build_merged_check_matrices(kind)
            )
            H_merged = H_X if check_type == "X" else H_Z
            labels = labels_X if check_type == "X" else labels_Z
            for patch, offset in (("A", 0), ("B", 12)):
                grown_row = (
                    geometry["grown_checks"][patch]["check_row"]
                    if check_type == grown_type
                    else None
                )
                for i in range(4):
                    if i == grown_row:
                        continue
                    label = f"{patch}.S{check_type}{i}"
                    row = H_merged[labels.index(label)]
                    assert np.array_equal(
                        row[offset : offset + 9], H_base[i]
                    )
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
        geometry = surgery.SEAM_GEOMETRIES[kind]
        through_logical = (
            geometry["merged_X_L"] if kind == "ZZ" else geometry["merged_Z_L"]
        )
        seam_crossings = [e for e in through_logical if e[0] == "S"]
        assert seam_crossings == [("S", 0)]

        # Canonical conversion: for ZZ, X2X4X6(B) = leftcol(B) * SX0 * SX1
        # where SX1 is B's grown check; for XX, Z0Z4Z8(A) = toprow(A) *
        # SZ1 * SZ3 where SZ3 is A's grown check.
        grown_in_conversion = "B" if kind == "ZZ" else "A"
        pickup = set()
        for _, si in geometry["grown_checks"][grown_in_conversion][
            "seam_pair"
        ]:
            pickup ^= {si}
        expected = {0} ^ pickup
        assert set(geometry["byproduct_seam_indices"]) == expected

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_telescope_reference_conversion(self, kind):
        """Declared reference checks convert telescoped to canonical reps.

        For each patch, (telescoped half) XOR (canonical logical) must equal
        the GF(2) sum of the declared telescope_reference_checks rows of the
        seam-check-type H matrix.
        """
        geometry = surgery.SEAM_GEOMETRIES[kind]
        H = surgery.BASE_H_Z if kind == "ZZ" else surgery.BASE_H_X
        canonical = np.zeros(9, dtype=int)
        for i in [0, 4, 8] if kind == "ZZ" else [2, 4, 6]:
            canonical[i] = 1
        for patch in ["A", "B"]:
            half = np.zeros(9, dtype=int)
            for p, i in geometry["parity_support"]:
                if p == patch:
                    half[i] = 1
            ref_sum = np.zeros(9, dtype=int)
            for row in geometry["telescope_reference_checks"][patch]:
                ref_sum ^= H[row]
            assert np.array_equal(half ^ canonical, ref_sum), (
                f"{kind} patch {patch}: telescope_reference_checks do not "
                "convert the telescoped half to the canonical logical"
            )


LAYOUTS = ["surf17", "surf13", "surf10"]


class TestSimplifiedSurgeryZZ:
    """Phase F: simplified (per-patch-decoded) ZZ lattice surgery."""

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("logical_x_on_l0", [False, True])
    def test_product_state_truth_table(self, layout, logical_x_on_l0):
        """|00> gives m_ZZ = 0; X_L on one patch gives 1, every shot.

        The post-surgery destructive FT Z measures also validate that the
        merge window leaves the per-patch Z decoding intact.
        """
        q0, q1, all_q = two_patch_setup(layout)
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", "L0", "L1", q0, q1, SEAMS, layout, mode="simple"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
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
        parities = collect_parities(results, "surgery_parity_zz")
        assert parities == [[expected]] * NUM_STIM_SHOTS
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_projective_and_consistent(self, layout):
        """|0+>: repeated M_ZZ agree per shot, ~50/50 across shots, and the
        destructive FT Z measure of the |+> patch equals the reported m."""
        q0, q1, all_q = two_patch_setup(layout)
        zz_insts = [
            surgery.build_surgery_parity_instruction(
                "ZZ", "L0", "L1", q0, q1, SEAMS, layout, mode="simple"
            )
            for _ in range(2)
        ]
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            ("Zero Prep", "L0"),
            ("Plus Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (zz_insts[0], None),
            (zz_insts[1], None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical Z Measure", "L0"),
            (
                "FT Logical Z Measure",
                "L1",
                (),
                {"reference_round_Z": True},  # |+> prep: Z round 0 is random
            ),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        pairs = collect_parities(results, "surgery_parity_zz")
        assert all(len(p) == 2 and p[0] == p[1] for p in pairs)
        ms = [p[0] for p in pairs]
        assert 0 < sum(ms) < NUM_STIM_SHOTS  # both branches appear
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        for shot, m in zip(logicals, ms):
            assert shot == [0, m]

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_x_correlation_preserved(self, layout):
        """|++> -> M_ZZ -> FT X measures XOR to 0 every shot.

        This exercises the split byproduct injection AND the grown-check
        history rewrite: any bookkeeping error shows up as a random XOR.
        Reference rounds are required in the grown-check (X) basis after a
        ZZ surgery (the rewrite lands in the round-0 layer).
        """
        q0, q1, all_q = two_patch_setup(layout)
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", "L0", "L1", q0, q1, SEAMS, layout, mode="simple"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            ("Plus Prep", "L0"),
            ("Plus Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (zz, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical X Measure", "L0", (), {"reference_round_X": True}),
            ("FT Logical X Measure", "L1", (), {"reference_round_X": True}),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        ms = [p[0] for p in collect_parities(results, "surgery_parity_zz")]
        assert 0 < sum(ms) < NUM_STIM_SHOTS  # |++> gives a random m_ZZ
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(shot[0] ^ shot[1] == 0 for shot in logicals)


class TestSimplifiedSurgeryXX:
    """Phase F: simplified XX lattice surgery (dual of the ZZ tests)."""

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("logical_z_on_l0", [False, True])
    def test_product_state_truth_table(self, layout, logical_z_on_l0):
        """|++> gives m_XX = 0; Z_L on one patch gives 1, every shot."""
        q0, q1, all_q = two_patch_setup(layout)
        xx = surgery.build_surgery_parity_instruction(
            "XX", "L0", "L1", q0, q1, SEAMS, layout, mode="simple"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
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
        parities = collect_parities(results, "surgery_parity_xx")
        assert parities == [[expected]] * NUM_STIM_SHOTS
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_projective_and_consistent(self, layout):
        """|+0>: repeated M_XX agree per shot; FT X of the |0> patch == m."""
        q0, q1, all_q = two_patch_setup(layout)
        xx_insts = [
            surgery.build_surgery_parity_instruction(
                "XX", "L0", "L1", q0, q1, SEAMS, layout, mode="simple"
            )
            for _ in range(2)
        ]
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            ("Plus Prep", "L0"),
            ("Zero Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (xx_insts[0], None),
            (xx_insts[1], None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical X Measure", "L0"),
            (
                "FT Logical X Measure",
                "L1",
                (),
                {"reference_round_X": True},  # |0> prep: X round 0 is random
            ),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        pairs = collect_parities(results, "surgery_parity_xx")
        assert all(len(p) == 2 and p[0] == p[1] for p in pairs)
        ms = [p[0] for p in pairs]
        assert 0 < sum(ms) < NUM_STIM_SHOTS
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        for shot, m in zip(logicals, ms):
            assert shot == [0, m]

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_z_correlation_preserved(self, layout):
        """|00> -> M_XX -> FT Z measures XOR to 0 every shot (dual of the
        ZZ x-correlation test; reference rounds in the grown Z basis)."""
        q0, q1, all_q = two_patch_setup(layout)
        xx = surgery.build_surgery_parity_instruction(
            "XX", "L0", "L1", q0, q1, SEAMS, layout, mode="simple"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            ("Zero Prep", "L0"),
            ("Zero Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (xx, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            ("FT Logical Z Measure", "L0", (), {"reference_round_Z": True}),
            ("FT Logical Z Measure", "L1", (), {"reference_round_Z": True}),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        ms = [p[0] for p in collect_parities(results, "surgery_parity_xx")]
        assert 0 < sum(ms) < NUM_STIM_SHOTS  # |00> gives a random m_XX
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(shot[0] ^ shot[1] == 0 for shot in logicals)


class TestSimplifiedSurgeryBell:
    """Phase F: surgery ZZ and XX parities on a Bell pair in one shot."""

    @pytest.mark.parametrize("layout", LAYOUTS)
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
        seams_v = ["SV0", "SV1", "SV2"]
        seams_h = ["SH0", "SH1", "SH2"]
        all_q = q0 + q1 + seams_v + seams_h
        cnot = multipatch.build_transversal_cnot_instruction(
            "L0", "L1", q0[:9], q1[:9]
        )
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", "L0", "L1", q0, q1, seams_v, layout, mode="simple"
        )
        xx = surgery.build_surgery_parity_instruction(
            "XX", "L0", "L1", q0, q1, seams_h, layout, mode="simple"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
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
        assert collect_parities(results, "surgery_parity_zz") == (
            [[0]] * NUM_STIM_SHOTS
        )
        assert collect_parities(results, "surgery_parity_xx") == (
            [[0]] * NUM_STIM_SHOTS
        )


class TestFTSurgery:
    """Phase G: FT surgery through the merged-matching-graph window decode."""

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("logical_x_on_l0", [False, True])
    def test_zz_truth_table_ft(self, layout, logical_x_on_l0):
        """FT-mode ZZ truth table: |00> -> 0, X_L(L0) -> 1, every shot."""
        q0, q1, all_q = two_patch_setup(layout)
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", "L0", "L1", q0, q1, SEAMS, layout, mode="ft"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
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
        assert collect_parities(results, "surgery_parity_zz") == (
            [[expected]] * NUM_STIM_SHOTS
        )
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("logical_z_on_l0", [False, True])
    def test_xx_truth_table_ft(self, layout, logical_z_on_l0):
        """FT-mode XX truth table: |++> -> 0, Z_L(L0) -> 1, every shot."""
        q0, q1, all_q = two_patch_setup(layout)
        xx = surgery.build_surgery_parity_instruction(
            "XX", "L0", "L1", q0, q1, SEAMS, layout, mode="ft"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
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
        assert collect_parities(results, "surgery_parity_xx") == (
            [[expected]] * NUM_STIM_SHOTS
        )
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert logicals == [[expected, 0]] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_bell_zz_then_xx_ft(self, layout):
        """Bell pair: FT-mode surgery M_ZZ and M_XX both report 0."""
        q0 = layout_qubits(layout, "_0")
        q1 = layout_qubits(layout, "_1")
        seams_v = ["SV0", "SV1", "SV2"]
        seams_h = ["SH0", "SH1", "SH2"]
        all_q = q0 + q1 + seams_v + seams_h
        cnot = multipatch.build_transversal_cnot_instruction(
            "L0", "L1", q0[:9], q1[:9]
        )
        zz = surgery.build_surgery_parity_instruction(
            "ZZ", "L0", "L1", q0, q1, seams_v, layout, mode="ft"
        )
        xx = surgery.build_surgery_parity_instruction(
            "XX", "L0", "L1", q0, q1, seams_h, layout, mode="ft"
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
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
        assert collect_parities(results, "surgery_parity_zz") == (
            [[0]] * NUM_STIM_SHOTS
        )
        assert collect_parities(results, "surgery_parity_xx") == (
            [[0]] * NUM_STIM_SHOTS
        )

    @pytest.mark.parametrize("kind", ["ZZ", "XX"])
    def test_weight1_injection_sweep(self, kind):
        """Every weight-1 Pauli fault in the merge window is tolerated.

        Sweeps Gxpi/Gypi/Gzpi before every component of the seam prep, all
        three merge-SE rounds, and the seam measurement (surf17). Joint
        parity and both destructive logical measures must all stay 0. This
        is the merged-graph validation (plan test group 6); the XX sweep on
        |++> additionally exercises Z-ancilla hook propagation in the
        X-error sector.
        """
        layout = "surf17"
        q0, q1, all_q = two_patch_setup(layout)
        seq = surgery.build_surgery_parity_instruction_sequence(
            kind, "L0", "L1", q0, q1, SEAMS, layout, mode="ft"
        )
        prep = "Zero Prep" if kind == "ZZ" else "Plus Prep"
        meas = "FT Logical Z Measure" if kind == "ZZ" else "FT Logical X Measure"
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            (prep, "L0"),
            (prep, "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (seq[0], None),   # 7: seam prep
            (seq[1], None),   # 8: merge SE round 1
            (seq[2], None),   # 9: merge SE round 2
            (seq[3], None),   # 10: merge SE round 3
            (seq[4], None),   # 11: merge bookkeeping
            (seq[5], None),   # 12: seam measurement
            (seq[6], None),   # 13: split bookkeeping
            ("QEC", "L0"),
            ("QEC", "L1"),
            (meas, "L0"),
            (meas, "L1"),
        ]
        base_program = make_stim_program(layout, stack, all_q)
        parity_key = f"surgery_parity_{kind.lower()}"
        sweep_targets = [
            (seq[0], 7),   # seam prep
            (seq[1], 8),   # SE round 1
            (seq[2], 9),   # SE round 2
            (seq[3], 10),  # SE round 3
            (seq[5], 12),  # seam measurement
        ]
        for inst, idx in sweep_targets:
            injected = fttools.build_discrete_error_injection_programs(
                base_program=base_program,
                instruction_to_analyze=inst,
                stack_idx_to_modify=idx,
                error_circuit_labels=["Gxpi", "Gypi", "Gzpi"],
            )
            failed = fttools.run_discrete_error_injected_programs(
                injected,
                [(parity_key, "all", True), ("logical_measurement", "all", True)],
                [[0], [0, 0]],
            )
            assert not failed, (
                f"{kind} stack idx {idx}: "
                + "; ".join(
                    f.name.split("+ injected error ")[-1] for f in failed
                )
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
        """A random-basis: readout_A ^ readout_B == m every shot, m random."""
        layout = "surf17"
        q0, q1, all_q = two_patch_setup(layout)
        inst = surgery.build_surgery_parity_instruction(
            kind, "L0", "L1", q0, q1, SEAMS, layout, mode=mode
        )
        if kind == "ZZ":
            prep0, prep1 = "Plus Prep", "Zero Prep"
            meas = "FT Logical Z Measure"
            flag = {"reference_round_Z": True}
        else:
            prep0, prep1 = "Zero Prep", "Plus Prep"
            meas = "FT Logical X Measure"
            flag = {"reference_round_X": True}
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            (prep0, "L0"),
            (prep1, "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (inst, None),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (meas, "L0", (), flag),
            (meas, "L1"),
        ]
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        parities = [
            p[0]
            for p in collect_parities(results, f"surgery_parity_{kind.lower()}")
        ]
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(
            (logicals[i][0] ^ logicals[i][1]) == parities[i]
            for i in range(NUM_STIM_SHOTS)
        )
        # The parity itself is a fair coin for these product states.
        assert 0 < sum(parities) < NUM_STIM_SHOTS


class TestSurgeryCnot:
    """Phase H: lattice-surgery CNOT (M_ZZ . M_XX through a |+> ancilla).

    L-shaped 3-patch layout: control C above ancilla ANC (vertical ZZ seam),
    target T right of ANC (horizontal XX seam). Verifies the correction
    table Z_L(C)^m_xx, X_L(T)^(m_zz ^ m_anc) via both truth tables and Bell
    correlations (plan test group 7). surf17, FT mode, 57 qubits: stim only.
    """

    @staticmethod
    def _cnot_setup():
        qc = layout_qubits("surf17", "_c")
        qt = layout_qubits("surf17", "_t")
        qa = layout_qubits("surf17", "_a")
        seams_v = ["SV0", "SV1", "SV2"]
        seams_h = ["SH0", "SH1", "SH2"]
        all_q = qc + qt + qa + seams_v + seams_h
        seq = surgery.build_surgery_cnot_sequence(
            "C", "T", "ANC", qc, qt, qa, seams_v, seams_h, "surf17", mode="ft"
        )
        prelude = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("C", qc)),
            ("Init Patch SURF", None, ("T", qt)),
            ("Init Patch SURF", None, ("ANC", qa)),
        ]
        return prelude, seq, all_q

    @pytest.mark.parametrize(
        "flips, expected",
        [
            ((), [0, 0]),
            (("C",), [1, 1]),
            (("T",), [0, 1]),
            (("C", "T"), [1, 0]),
        ],
        ids=["00", "XC", "XT", "XCXT"],
    )
    def test_cnot_truth_table_z(self, flips, expected):
        """Z-basis truth table: X_L(C) propagates to T, X_L(T) unchanged."""
        prelude, seq, all_q = self._cnot_setup()
        stack = prelude + [
            ("Zero Prep", "C"),
            ("Zero Prep", "T"),
            *[("X", lbl) for lbl in flips],
            ("QEC", "C"),
            ("QEC", "T"),
            *seq,
            ("QEC", "C"),
            ("QEC", "T"),
            # T's XX merge grows a Z check -> reference_round_Z on T.
            ("FT Logical Z Measure", "C"),
            ("FT Logical Z Measure", "T", (), {"reference_round_Z": True}),
        ]
        program = make_stim_program("surf17", stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        # Per-shot entries: [m_anc, m_C, m_T].
        assert [lg[1:] for lg in logicals] == [expected] * NUM_STIM_SHOTS

    @pytest.mark.parametrize(
        "flips, expected",
        [
            ((), [0, 0]),
            (("C",), [1, 0]),
            (("T",), [1, 1]),
        ],
        ids=["plusplus", "ZC", "ZT"],
    )
    def test_cnot_truth_table_x(self, flips, expected):
        """X-basis truth table: Z_L(T) kicks back to C, Z_L(C) unchanged."""
        prelude, seq, all_q = self._cnot_setup()
        stack = prelude + [
            ("Plus Prep", "C"),
            ("Plus Prep", "T"),
            *[("Z", lbl) for lbl in flips],
            ("QEC", "C"),
            ("QEC", "T"),
            *seq,
            ("QEC", "C"),
            ("QEC", "T"),
            # C's ZZ merge grows an X check -> reference_round_X on C.
            ("FT Logical X Measure", "C", (), {"reference_round_X": True}),
            ("FT Logical X Measure", "T"),
        ]
        program = make_stim_program("surf17", stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert [lg[1:] for lg in logicals] == [expected] * NUM_STIM_SHOTS

    @pytest.mark.parametrize("basis", ["Z", "X"])
    def test_cnot_bell_correlations(self, basis):
        """|+>_C |0>_T -> Bell: perfect ZZ and XX correlations, random
        marginals.

        Reference flags per patch combine the surgery-induced rule (C FT X /
        T FT Z after the grown-check rewrites) with the prep-conjugate rule
        (C prepped |+> -> FT Z; T prepped |0> -> FT X).
        """
        prelude, seq, all_q = self._cnot_setup()
        flag = {f"reference_round_{basis}": True}
        stack = prelude + [
            ("Plus Prep", "C"),
            ("Zero Prep", "T"),
            ("QEC", "C"),
            ("QEC", "T"),
            *seq,
            ("QEC", "C"),
            ("QEC", "T"),
            (f"FT Logical {basis} Measure", "C", (), flag),
            (f"FT Logical {basis} Measure", "T", (), flag),
        ]
        program = make_stim_program("surf17", stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(lg[1] ^ lg[2] == 0 for lg in logicals)
        outcomes = [lg[1] for lg in logicals]
        assert 0 < sum(outcomes) < NUM_STIM_SHOTS


class TestSurgeryDenseSmoke:
    """Phase F: statevector smoke on the intended dense configuration."""

    def test_zz_statevector_surf10(self):
        """2x surf10 + 3 seams (23 qubits): |00> -> m_ZZ = 0."""
        layout = "surf10"
        q0, q1, all_q = two_patch_setup(layout)
        # 2 merge rounds keep the 23-qubit statevector runtime tolerable;
        # noiseless round parities are constant so the vote is unaffected.
        zz = surgery.build_surgery_parity_instruction(
            "ZZ",
            "L0",
            "L1",
            q0,
            q1,
            SEAMS,
            layout,
            mode="simple",
            num_merge_rounds=2,
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            ("Zero Prep", "L0"),
            ("Zero Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (zz, None),
        ]
        code = codepack_surf17.create_qec_code(layout=layout, num_qec_rounds=2)
        model = codepack_surf17.create_ideal_model(
            all_q,
            gaterep=GateRep.UNITARY,
            model_backend=DictNoiseModel,
        )
        program = QuantumProgram(
            stack,
            default_noise_model=model,
            state_type=NumpyStatevectorQuantumState,
            patch_types={"SURF": code},
        )
        results = program.run(num_shots=NUM_KRAUS_SHOTS, verbose=False)
        parities = collect_parities(results, "surgery_parity_zz")
        assert parities == [[0]] * NUM_KRAUS_SHOTS
