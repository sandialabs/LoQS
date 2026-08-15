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
    DictNoiseModel,
    NumpyStatevectorQuantumState,
    STIMQuantumState,
    StimCircuitGateRep,
    UnitaryGateRep,
)
from loqs.backends.circuit.pygsticircuit import PyGSTiPhysicalCircuit
from loqs.core import QuantumProgram
from loqs.core.instructions import builders
from loqs.codepacks import codepack_surf17_surgery as surgery
from loqs.codepacks import codepack_surf17_multipatch as multipatch
from loqs.codepacks import codepack_surf17_tomita2014 as codepack_surf17
from loqs.tools import fttools

NUM_STIM_SHOTS = 100
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
        stack,
        default_noise_model=model,
        state_type=STIMQuantumState,
        patch_types={"SURF": code},
    )


SEAMS = ["Qs0", "Qs1", "Qs2"]


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
                {"reference_round_mode_Z": "guarded_diff"},  # |+> prep: Z round 0 is random
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
            ("FT Logical X Measure", "L0", (), {"reference_round_mode_X": "guarded_diff"}),
            ("FT Logical X Measure", "L1", (), {"reference_round_mode_X": "guarded_diff"}),
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
                {"reference_round_mode_X": "guarded_diff"},  # |0> prep: X round 0 is random
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
            ("FT Logical Z Measure", "L0", (), {"reference_round_mode_Z": "guarded_diff"}),
            ("FT Logical Z Measure", "L1", (), {"reference_round_mode_Z": "guarded_diff"}),
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
        seams_v = ["Qsv0", "Qsv1", "Qsv2"]
        seams_h = ["Qsh0", "Qsh1", "Qsh2"]
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
        seams_v = ["Qsv0", "Qsv1", "Qsv2"]
        seams_h = ["Qsh0", "Qsh1", "Qsh2"]
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

    @pytest.mark.slow
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
            flag = {"reference_round_mode_Z": "guarded_diff"}
        else:
            prep0, prep1 = "Zero Prep", "Plus Prep"
            meas = "FT Logical X Measure"
            flag = {"reference_round_mode_X": "guarded_diff"}
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

    L-shaped 3-patch layout: control C above ancilla Qanc (vertical ZZ seam),
    target T right of Qanc (horizontal XX seam). Verifies the correction
    table Z_L(C)^m_xx, X_L(T)^(m_zz ^ m_anc) via both truth tables and Bell
    correlations (plan test group 7). surf17, FT mode, 57 qubits: stim only.
    """

    @staticmethod
    def _cnot_setup():
        qc = layout_qubits("surf17", "_c")
        qt = layout_qubits("surf17", "_t")
        qa = layout_qubits("surf17", "_a")
        seams_v = ["Qsv0", "Qsv1", "Qsv2"]
        seams_h = ["Qsh0", "Qsh1", "Qsh2"]
        all_q = qc + qt + qa + seams_v + seams_h
        seq = surgery.build_surgery_cnot_sequence(
            "C", "T", "Qanc", qc, qt, qa, seams_v, seams_h, "surf17", mode="ft"
        )
        prelude = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("C", qc)),
            ("Init Patch SURF", None, ("T", qt)),
            ("Init Patch SURF", None, ("Qanc", qa)),
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
            # T's XX merge grows a Z check -> reference_round_mode_Z="guarded_diff" on T.
            ("FT Logical Z Measure", "C"),
            ("FT Logical Z Measure", "T", (), {"reference_round_mode_Z": "guarded_diff"}),
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
            # C's ZZ merge grows an X check -> reference_round_mode_X="guarded_diff" on C.
            ("FT Logical X Measure", "C", (), {"reference_round_mode_X": "guarded_diff"}),
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
        flag = {f"reference_round_mode_{basis}": "guarded_diff"}
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

    @pytest.mark.slow
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
        parities = collect_parities(results, "surgery_parity_zz")
        assert parities == [[0]] * NUM_KRAUS_SHOTS


class TestMzzBellPrep:
    """Phase I: Bell prep by a direct M_ZZ merge on |+>_L |+>_L.

    build_mzz_bell_prep_sequence measures Z_L Z_L between the two patches
    (no ancilla patch) and injects the conditional frame correction
    X_L(L1)^m_zz, leaving (|00> + |11>)/sqrt(2). Verified by perfect
    XOR-parity in destructive FT readouts of BOTH bases while m_zz itself
    stays random (~50/50).
    """

    @staticmethod
    def bell_prep_stack(layout, q0, q1, all_q, mode, basis):
        """|+>|+> -> M_ZZ Bell prep -> QEC -> FT readout of both patches."""
        seq = surgery.build_mzz_bell_prep_sequence(
            "L0", "L1", q0, q1, SEAMS, layout, mode=mode
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
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
            (f"FT Logical {basis} Measure", "L0", (), dict(flag)),
            (f"FT Logical {basis} Measure", "L1", (), dict(flag)),
        ]
        return stack

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("mode", ["simple", "ft"])
    def test_bell_zz_correlation(self, layout, mode):
        """FT Z readouts XOR to 0 every shot while m_zz is random.

        Without the conditional X_L correction the XOR would equal m_zz,
        so this proves the correction fires on exactly the m_zz = 1 shots.
        """
        q0, q1, all_q = two_patch_setup(layout)
        stack = self.bell_prep_stack(layout, q0, q1, all_q, mode, "Z")
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        ms = [p[0] for p in collect_parities(results, "surgery_parity_zz")]
        assert 0 < sum(ms) < NUM_STIM_SHOTS  # both branches appear
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(shot[0] ^ shot[1] == 0 for shot in logicals)

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("mode", ["simple", "ft"])
    def test_bell_xx_correlation(self, layout, mode):
        """FT X readouts XOR to 0 every shot (X_L frame is invisible in X)."""
        q0, q1, all_q = two_patch_setup(layout)
        stack = self.bell_prep_stack(layout, q0, q1, all_q, mode, "X")
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        ms = [p[0] for p in collect_parities(results, "surgery_parity_zz")]
        assert 0 < sum(ms) < NUM_STIM_SHOTS
        logicals = results.collect_shot_data(
            "logical_measurement", "all", strip_none_entries=True
        )
        assert all(shot[0] ^ shot[1] == 0 for shot in logicals)

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("mode", ["simple", "ft"])
    def test_correction_matches_parity(self, layout, mode):
        """The recorded mzz_bell_correction equals that shot's m_zz."""
        q0, q1, all_q = two_patch_setup(layout)
        stack = self.bell_prep_stack(layout, q0, q1, all_q, mode, "Z")
        program = make_stim_program(layout, stack, all_q)
        results = program.run(num_shots=NUM_STIM_SHOTS, verbose=False)
        ms = [p[0] for p in collect_parities(results, "surgery_parity_zz")]
        corrections = results.collect_shot_data(
            "mzz_bell_correction", "all", strip_none_entries=True
        )
        assert [c[0] for c in corrections] == ms


class TestMzzFaultTolerance:
    """Phase J: single-fault sweep of the mzz Bell-prep merge window.

    Injects discrete Pauli faults into the mzz-specific circuitry (seam
    prep, the three merged-SE rounds, seam measurement) inside the FULL
    mzz Bell-prep protocol on |+>_L |+>_L, for all three layouts. The
    per-patch prep/QEC/readout stages are already covered by the
    tomita2014 FT tests; this sweep targets the merge window, where
    surf13/surf10 reuse shared (surf13) or a single (surf10) borrowed
    ancilla for the seam checks — the highest hook-error risk.

    Pass criterion: m_zz is legitimately random on |+>|+>, so the FT
    invariant is per-shot agreement of the two decoded logical readouts,
    `logical_measurement[0] ^ logical_measurement[1] == 0`, in both
    terminating bases — NOT a fixed expected outcome. Every injected
    fault also stress-tests the downstream chain (reference-round-flag
    decode, conditional frame correction, per-patch termination decode).

    Fault models: weight-1 Paulis before every circuit component, and
    all 9 correlated 2-qubit Pauli pairs after every CNOT of the
    merged-SE rounds (`post_twoq_gates=True`). `ft` decode must show
    zero failures; `simple` decode is characterized (xfail with counts)
    rather than asserted.
    """

    pytestmark = pytest.mark.slow

    WEIGHT1_LABELS = ["Gxpi", "Gypi", "Gzpi"]
    # seq[i] sits at stack index 7 + i in _mzz_program's stack.
    WEIGHT1_SEQ_IDXS = (0, 1, 2, 3, 5)  # seam prep, SE x3, seam measure
    POST2Q_SEQ_IDXS = (1, 2, 3)  # only the SE rounds contain 2q gates

    @staticmethod
    def _mzz_program(layout, mode, basis):
        """(base_program, seq): full mzz Bell prep with the merge window
        as individually injectable stack entries 7..13. In ft mode the
        post-split byproduct repair runs after the post-merge QEC (the
        middle-seam fix; simple mode stays unrepaired by design)."""
        q0, q1, all_q = two_patch_setup(layout)
        seq = surgery.build_surgery_parity_instruction_sequence(
            "ZZ", "L0", "L1", q0, q1, SEAMS, layout, mode=mode
        )
        corrections = surgery.build_mzz_bell_corrections_instruction(
            "L1", q1[:9]
        )
        # Z round 0 is random after |+> prep -> guarded_diff. X round 0
        # is deterministic -> kept as a detector layer ("raw", the split
        # bookkeeping's round-0 offset absorbs the grown-check rewrite),
        # closing the prep blind window.
        flag = {
            f"reference_round_mode_{basis}": (
                "guarded_diff" if basis == "Z" else "raw"
            )
        }
        meas = f"FT Logical {basis} Measure"
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("L0", q0)),
            ("Init Patch SURF", None, ("L1", q1)),
            ("Plus Prep", "L0"),
            ("Plus Prep", "L1"),
            ("QEC", "L0"),
            ("QEC", "L1"),
            (seq[0], None),  # 7: seam prep
            (seq[1], None),  # 8: merge SE round 1
            (seq[2], None),  # 9: merge SE round 2
            (seq[3], None),  # 10: merge SE round 3
            (seq[4], None),  # 11: merge bookkeeping
            (seq[5], None),  # 12: seam measurement
            (seq[6], None),  # 13: split bookkeeping
            (corrections, None),  # 14: conditional frame X_L(L1)^m_zz
            ("QEC", "L0"),
            ("QEC", "L1"),
        ]
        if mode == "ft":
            repair = surgery.build_split_byproduct_repair_instruction(
                "ZZ", "L0", "L1", q0, q1, SEAMS, layout,
                num_post_split_rounds=3,
            )
            stack.append((repair, None))
        stack += [
            (meas, "L0", (), dict(flag)),
            (meas, "L1", (), dict(flag)),
        ]
        return make_stim_program(layout, stack, all_q), seq

    @staticmethod
    def _run_xor_sweep(programs, num_shots=1):
        """Run injected programs; return those where any shot's decoded
        logical readouts disagree (XOR != 0). 1 shot/program leaves
        P(miss the m_zz = 1 branch) = 1/2 per program; failures are
        reconfirmed at 32 shots by the callers, so this only risks a
        (per-run, not systematic) false negative for a bug that manifests
        on the random branch at exactly one specific location."""
        failed = []
        for program in programs:
            results = program.run(num_shots=num_shots, verbose=False)
            logicals = results.collect_shot_data(
                "logical_measurement", "all", strip_none_entries=True
            )
            if any(shot[0] ^ shot[1] for shot in logicals):
                failed.append(program)
        return failed

    def _sweep(self, layout, mode, basis, post_twoq):
        """(failed_programs, total_injected) over the merge window.

        "total_injected" counts the Pauli-propagation equivalence-class
        representatives actually run, not the raw location x label
        count -- see [](api:fttools.build_pruned_discrete_error_injection_programs)."""
        base_program, seq = self._mzz_program(layout, mode, basis)
        seq_idxs = self.POST2Q_SEQ_IDXS if post_twoq else self.WEIGHT1_SEQ_IDXS
        failed, total = [], 0
        for i in seq_idxs:
            injected, _ = fttools.build_pruned_discrete_error_injection_programs(
                base_program=base_program,
                instruction_to_analyze=seq[i],
                stack_idx_to_modify=7 + i,
                error_circuit_labels=self.WEIGHT1_LABELS,
                post_twoq_gates=post_twoq,
            )
            total += len(injected)
            failed += self._run_xor_sweep(injected)
        return failed, total

    @staticmethod
    def _error_tags(programs, limit=5):
        return [
            p.name.split("+ injected error ")[-1] for p in programs[:limit]
        ]

    def _assert_ft(self, layout, basis, post_twoq):
        failed, total = self._sweep(layout, "ft", basis, post_twoq)
        if failed:  # reconfirm before failing: kill statistical flukes
            failed = self._run_xor_sweep(failed, num_shots=32)
        assert not failed, (
            f"{layout}/{basis}: {len(failed)}/{total} injected faults broke "
            f"the XOR invariant, e.g. {self._error_tags(failed)}"
        )

    def _characterize_simple(self, layout, basis, post_twoq):
        failed, total = self._sweep(layout, "simple", basis, post_twoq)
        if failed:
            failed = self._run_xor_sweep(failed, num_shots=32)
        if failed:
            pytest.xfail(
                f"simple decode {layout}/{basis}: {len(failed)}/{total} "
                f"failing locations, e.g. {self._error_tags(failed)}"
            )

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_weight1_sweep_ft(self, layout, basis):
        """No weight-1 fault in the merge window flips the witness bit."""
        self._assert_ft(layout, basis, post_twoq=False)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_post2q_sweep_ft(self, layout, basis):
        """No correlated 2q Pauli pair after any merged-SE CNOT flips it."""
        self._assert_ft(layout, basis, post_twoq=True)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_weight1_sweep_simple(self, layout, basis):
        """Characterize the simple (non-FT window) decode under weight-1."""
        self._characterize_simple(layout, basis, post_twoq=False)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_post2q_sweep_simple(self, layout, basis):
        """Characterize the simple decode under post-2Q correlated pairs."""
        self._characterize_simple(layout, basis, post_twoq=True)


class TestSurgeryCnotFaultTolerance:
    """Phase K: single-fault sweep of the surgery-CNOT merge windows.

    Injects discrete Pauli faults into the surgery-specific circuitry of
    BOTH merges (seam prep, three merged-SE rounds, seam measurement of
    the ZZ merge ctrl-anc and the XX merge anc-tgt) inside the full
    Bell-prep protocol |+>_C |0>_T -> CNOT -> Bell, for all three
    layouts. Per-patch prep/QEC/readout stages are covered by the
    tomita2014 FT tests; the mzz sweep (Phase J) covers a lone ZZ merge
    terminated in matching bases. This sweep exercises the CNOT-specific
    plumbing: byproduct flips reaching the logical outcome through m_xx
    (ZZ byproduct Z_L on the ancilla flips the X_L(anc) reading) and
    through the X_L(tgt) correction (XX byproduct), plus the ancilla's
    destructive Z decode feeding m_anc.

    ft mode includes the split-byproduct repairs wired into
    build_surgery_cnot_sequence: fire_rule="b_only" after the ZZ merge
    (the ancilla, its patch B, gets no byproduct-sensitive termination
    decode, so the UNCOMPENSATED single-fault class is the anc-side
    outer seam qubit, not the middle one) and fire_rule="both" after
    the XX merge (anc compensates via m_anc, tgt via its own Z decode;
    the middle seam qubit is the broken class - the mzz dual). Both
    classes were confirmed empirically before the fix.

    Pass criterion: per-shot XOR of the decoded C and T logical
    readouts == 0 in both terminating bases (deterministic for a Bell
    pair; m_zz/m_xx/m_anc are legitimately random). ft decode must show
    zero failures; simple decode is characterized (xfail with counts).
    """

    pytestmark = pytest.mark.slow

    WEIGHT1_LABELS = ["Gxpi", "Gypi", "Gzpi"]
    WEIGHT1_SEQ_IDXS = (0, 1, 2, 3, 5)  # seam prep, SE x3, seam measure
    POST2Q_SEQ_IDXS = (1, 2, 3)  # only the SE rounds contain 2q gates

    @staticmethod
    def _cnot_program(layout, mode, basis):
        """(program, targets): surgery-CNOT Bell prep with both merge
        windows as individually injectable stack entries. targets maps
        merge kind -> (seq, base stack index of seq[0]). Mirrors
        build_surgery_cnot_sequence (including repair placement) but
        built from the flat parity sequences for injectability."""
        qc = layout_qubits(layout, "_c")
        qt = layout_qubits(layout, "_t")
        qa = layout_qubits(layout, "_a")
        seams_v = ["Qsv0", "Qsv1", "Qsv2"]
        seams_h = ["Qsh0", "Qsh1", "Qsh2"]
        all_q = qc + qt + qa + seams_v + seams_h
        zzseq = surgery.build_surgery_parity_instruction_sequence(
            "ZZ", "C", "ANC", qc, qa, seams_v, layout, mode=mode
        )
        xxseq = surgery.build_surgery_parity_instruction_sequence(
            "XX", "ANC", "T", qa, qt, seams_h, layout, mode=mode
        )
        corrections = surgery.build_surgery_cnot_corrections_instruction(
            "C", "T", qc, qt
        )
        stack = [
            ("Init State", None, (len(all_q),), {"qubit_labels": all_q}),
            ("Init Patch SURF", None, ("C", qc)),
            ("Init Patch SURF", None, ("T", qt)),
            ("Init Patch SURF", None, ("ANC", qa)),
            ("Plus Prep", "C"),
            ("Zero Prep", "T"),
            ("QEC", "C"),
            ("QEC", "T"),
            ("Plus Prep", "ANC"),
            ("QEC", "ANC"),
        ]
        zz_base = len(stack)
        stack += [(inst, None) for inst in zzseq]
        stack += [("QEC", "C"), ("QEC", "ANC")]
        if mode == "ft":
            stack.append(
                (
                    surgery.build_split_byproduct_repair_instruction(
                        "ZZ", "C", "ANC", qc, qa, seams_v, layout,
                        fire_rule="b_only",
                    ),
                    None,
                )
            )
        stack.append(("QEC", "T"))
        xx_base = len(stack)
        stack += [(inst, None) for inst in xxseq]
        stack.append(("QEC", "ANC"))
        if mode == "ft":
            stack += [
                ("QEC", "T"),
                (
                    surgery.build_split_byproduct_repair_instruction(
                        "XX", "ANC", "T", qa, qt, seams_h, layout
                    ),
                    None,
                ),
            ]
        flag = {f"reference_round_mode_{basis}": "guarded_diff"}
        stack += [
            ("FT Logical Z Measure", "ANC", (), {"reference_round_mode_Z": "guarded_diff"}),
            (corrections, None),
            ("QEC", "C"),
            ("QEC", "T"),
            (f"FT Logical {basis} Measure", "C", (), dict(flag)),
            (f"FT Logical {basis} Measure", "T", (), dict(flag)),
        ]
        targets = {"ZZ": (zzseq, zz_base), "XX": (xxseq, xx_base)}
        return make_stim_program(layout, stack, all_q), targets

    @staticmethod
    def _run_xor_sweep(programs, num_shots=1):
        """Run injected programs; return those where any shot's decoded
        C and T readouts disagree. Per-shot entries: [m_anc, m_C, m_T].
        1 shot/program; failures are reconfirmed at 32 shots by the
        callers, so this only risks a (per-run, not systematic) false
        negative for a bug that manifests on the random branch at
        exactly one specific location."""
        failed = []
        for program in programs:
            results = program.run(num_shots=num_shots, verbose=False)
            logicals = results.collect_shot_data(
                "logical_measurement", "all", strip_none_entries=True
            )
            if any(shot[1] ^ shot[2] for shot in logicals):
                failed.append(program)
        return failed

    def _sweep(self, layout, mode, basis, post_twoq):
        """(failed_programs, total_injected) over BOTH merge windows.

        "total_injected" counts the Pauli-propagation equivalence-class
        representatives actually run, not the raw location x label
        count -- see [](api:fttools.build_pruned_discrete_error_injection_programs)."""
        base_program, targets = self._cnot_program(layout, mode, basis)
        seq_idxs = self.POST2Q_SEQ_IDXS if post_twoq else self.WEIGHT1_SEQ_IDXS
        failed, total = [], 0
        for seq, base in targets.values():
            for i in seq_idxs:
                injected, _ = fttools.build_pruned_discrete_error_injection_programs(
                    base_program=base_program,
                    instruction_to_analyze=seq[i],
                    stack_idx_to_modify=base + i,
                    error_circuit_labels=self.WEIGHT1_LABELS,
                    post_twoq_gates=post_twoq,
                )
                total += len(injected)
                failed += self._run_xor_sweep(injected)
        return failed, total

    @staticmethod
    def _error_tags(programs, limit=5):
        return [
            p.name.split("+ injected error ")[-1] for p in programs[:limit]
        ]

    def _assert_ft(self, layout, basis, post_twoq):
        failed, total = self._sweep(layout, "ft", basis, post_twoq)
        if failed:  # reconfirm before failing: kill statistical flukes
            failed = self._run_xor_sweep(failed, num_shots=32)
        assert not failed, (
            f"{layout}/{basis}: {len(failed)}/{total} injected faults broke "
            f"the XOR invariant, e.g. {self._error_tags(failed)}"
        )

    def _characterize_simple(self, layout, basis, post_twoq):
        failed, total = self._sweep(layout, "simple", basis, post_twoq)
        if failed:
            failed = self._run_xor_sweep(failed, num_shots=32)
        if failed:
            pytest.xfail(
                f"simple decode {layout}/{basis}: {len(failed)}/{total} "
                f"failing locations, e.g. {self._error_tags(failed)}"
            )

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_weight1_sweep_ft(self, layout, basis):
        """No weight-1 fault in either merge window flips the Bell XOR."""
        self._assert_ft(layout, basis, post_twoq=False)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_post2q_sweep_ft(self, layout, basis):
        """No correlated 2q Pauli pair after any merged-SE CNOT flips it."""
        self._assert_ft(layout, basis, post_twoq=True)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_weight1_sweep_simple(self, layout, basis):
        """Characterize the simple (non-FT window) decode under weight-1."""
        self._characterize_simple(layout, basis, post_twoq=False)

    @pytest.mark.parametrize("basis", ["Z", "X"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_post2q_sweep_simple(self, layout, basis):
        """Characterize the simple decode under post-2Q correlated pairs."""
        self._characterize_simple(layout, basis, post_twoq=True)


class TestMzzFaultToleranceSmoke:
    """Fast, non-exhaustive companion to `TestMzzFaultTolerance`.

    Samples a handful of weight-1 fault locations per merge-window
    component on the cheapest (surf10) layout instead of every location
    on all three layouts, so the default test run still catches gross
    breakage in the mzz-merge FT machinery without paying for the full
    sweep (marked `slow`; run it explicitly with `-m slow` for the
    exhaustive check).
    """

    SAMPLES_PER_LOCATION = 2
    # Subset of TestMzzFaultTolerance.WEIGHT1_SEQ_IDXS: seam prep (cheap,
    # 6 locations) plus one merged-SE round (the richest component, 164
    # locations, and where hook errors matter most). Building the full
    # per-location list (before sampling) is what costs time, so this
    # subset -- not just SAMPLES_PER_LOCATION -- is what keeps this fast.
    SEQ_IDXS = (0, 1)

    def test_weight1_smoke_surf10(self):
        """A handful of weight-1 faults per merge-window component are
        tolerated on surf10/Z, sampled from the exhaustive sweep's full
        location set."""
        failed, total = self._smoke_sweep("surf10", "ft", "Z")
        assert not failed, (
            f"{len(failed)}/{total} sampled fault locations broke the "
            f"XOR invariant, e.g. "
            f"{TestMzzFaultTolerance._error_tags(failed)}"
        )

    @classmethod
    def _smoke_sweep(cls, layout, mode, basis):
        base_program, seq = TestMzzFaultTolerance._mzz_program(
            layout, mode, basis
        )
        failed, total = [], 0
        for i in cls.SEQ_IDXS:
            injected = fttools.build_discrete_error_injection_programs(
                base_program=base_program,
                instruction_to_analyze=seq[i],
                stack_idx_to_modify=7 + i,
                error_circuit_labels=TestMzzFaultTolerance.WEIGHT1_LABELS,
            )
            if not injected:
                continue
            step = max(1, len(injected) // cls.SAMPLES_PER_LOCATION)
            sample = injected[::step][: cls.SAMPLES_PER_LOCATION]
            total += len(sample)
            failed += TestMzzFaultTolerance._run_xor_sweep(sample)
        return failed, total


class TestSurgeryCnotFaultToleranceSmoke:
    """Fast, non-exhaustive companion to `TestSurgeryCnotFaultTolerance`.

    Samples a handful of weight-1 fault locations per merge-window
    component on the cheapest (surf10) layout instead of the full
    ~3000-program sweep per layout/basis combination, so the default
    test run still catches gross breakage in the surgery-CNOT FT
    machinery without the exhaustive cost (marked `slow`; run it
    explicitly with `-m slow` for the exhaustive check).
    """

    SAMPLES_PER_LOCATION = 2
    # Subset of TestSurgeryCnotFaultTolerance.WEIGHT1_SEQ_IDXS: seam prep
    # (cheap, 6 locations) plus one merged-SE round (the richest
    # component, 164 locations). Building the full per-location list
    # (before sampling) is what costs time, so this subset -- not just
    # SAMPLES_PER_LOCATION -- is what keeps this fast.
    SEQ_IDXS = (0, 1)

    def test_weight1_smoke_surf10(self):
        """A handful of weight-1 faults per merge-window component are
        tolerated on surf10/Z, sampled from the exhaustive sweep's full
        location set."""
        failed, total = self._smoke_sweep("surf10", "ft", "Z")
        assert not failed, (
            f"{len(failed)}/{total} sampled fault locations broke the "
            f"Bell XOR invariant, e.g. "
            f"{TestSurgeryCnotFaultTolerance._error_tags(failed)}"
        )

    @classmethod
    def _smoke_sweep(cls, layout, mode, basis):
        base_program, targets = TestSurgeryCnotFaultTolerance._cnot_program(
            layout, mode, basis
        )
        failed, total = [], 0
        for seq, base in targets.values():
            for i in cls.SEQ_IDXS:
                injected = fttools.build_discrete_error_injection_programs(
                    base_program=base_program,
                    instruction_to_analyze=seq[i],
                    stack_idx_to_modify=base + i,
                    error_circuit_labels=(
                        TestSurgeryCnotFaultTolerance.WEIGHT1_LABELS
                    ),
                )
                if not injected:
                    continue
                step = max(1, len(injected) // cls.SAMPLES_PER_LOCATION)
                sample = injected[::step][: cls.SAMPLES_PER_LOCATION]
                total += len(sample)
                failed += TestSurgeryCnotFaultTolerance._run_xor_sweep(
                    sample
                )
        return failed, total
