"""Tester for loqs.backends.state.npsvstate.

Behaviour shared with the other state backends is in test_states.py.
"""

import numpy as np
import pytest

from loqs.backends.reps import (
    KrausGateRep,
    OutcomeOperationDictInstrumentRep,
    UnitaryGateRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.backends import NumpyStatevectorQuantumState as SVState


class TestNumPyStatevectorQuantumState:

    def _check(self, state, expected_state):
        assert state.qubit_labels == expected_state.qubit_labels
        assert np.allclose(state.state, expected_state.state)

    def test_init(self):
        qubit_labels = [f"Q{i}" for i in range(5)]
        s = SVState(5, qubit_labels)

        # Bitstring initializer
        s3 = SVState([0, 0, 0, 0, 0], qubit_labels)
        self._check(s3, s)

        # (Flat) Numpy array check also
        all0_state = np.zeros(2**5)
        all0_state[0] = 1
        s5 = SVState(all0_state, qubit_labels)
        self._check(s5, s)

        # Invalid initializer input raises
        with pytest.raises(ValueError):
            SVState("not a valid state")

    def test_apply_reps_inplace_unknown_rep_type_raises(self):
        """`apply_reps_inplace` must reject any rep that is neither a
        `GateRep` nor an `InstrumentRep` instance."""
        s = SVState([0], ["Q0"])
        with pytest.raises(ValueError):
            s.apply_reps_inplace([object()])

    def test_block_matvec_unknown_qubit_raises(self):
        s = SVState([0], ["Q0"], contraction="matmul")
        with pytest.raises(ValueError):
            s._block_matvec(np.eye(2), ["QBad"], s.state)

    @pytest.mark.parametrize("kraus_sampling", ["lazy", "choice"])
    def test_kraus_sampling_distribution(self, kraus_sampling):
        """Kraus channel sampling must match analytic Born-rule probabilities,
        and each outcome state must equal the normalized K_i |psi> for the
        branch taken. Written against behavior, not implementation, so both
        sampling modes are held to the same distributional contract."""

        # Asymmetric initial state so branch probabilities are state-dependent
        # and unequal -- catches interval/boundary and ordering bugs that a
        # symmetric 50/50 channel would mask
        theta = 0.4
        psi = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)

        def run_trials(psi, labels, reptuple, Ks, n_trials, seed):
            n_qubits = len(labels)
            # Analytic branch probabilities and post-application states
            probs_exact = [np.linalg.norm(K @ psi) ** 2 for K in Ks]
            posts_exact = [K @ psi / np.linalg.norm(K @ psi) for K in Ks]
            assert np.isclose(sum(probs_exact), 1)

            test = SVState(
                psi.copy().reshape((2,) * n_qubits),
                labels,
                seed=seed,
                kraus_sampling=kraus_sampling,
            )
            counts = [0] * len(Ks)
            for _ in range(n_trials):
                # manual reset, RNG stream continues
                test._state = psi.copy().reshape((2,) * n_qubits)
                test.apply_reps_inplace([reptuple])

                # Identify the branch by phase-insensitive overlap; the output
                # must *exactly* match that branch's normalized K_i |psi>
                out_flat = test.state.flatten()
                fids = [
                    np.abs(np.vdot(post, out_flat)) for post in posts_exact
                ]
                branch = int(np.argmax(fids))
                assert np.isclose(fids[branch], 1)
                counts[branch] += 1

            # Frequencies within 5 sigma (binomial) of Born probabilities --
            # deterministic given the seed, so no flakiness
            for count, p in zip(counts, probs_exact):
                sigma = np.sqrt(p * (1 - p) / n_trials)
                assert abs(count / n_trials - p) < 5 * sigma

        n_trials = 3_000
        labels = ["Q0"]

        # Case 1: amplitude damping with all probabilities None
        # (exercises the state-dependent probability path)
        gamma = 0.3
        A0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        A1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        amp_damp = KrausGateRep([(A0, None), (A1, None)], labels)
        run_trials(psi, labels, amp_damp, [A0, A1], n_trials, 20260708)

        # Case 2: three operators with a mix of given and None probabilities
        # (exercises mixed accumulation across the operator list)
        p_x, p_y = 0.2, 0.1
        U_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        U_Y = np.array([[0, -1j], [1j, 0]])
        K_I = np.sqrt(1 - p_x - p_y) * np.eye(2)
        K_X = np.sqrt(p_x) * U_X
        K_Y = np.sqrt(p_y) * U_Y
        mixed = KrausGateRep(
            [(K_I, 1 - p_x - p_y), (K_X, None), (K_Y, p_y)], labels
        )
        run_trials(psi, labels, mixed, [K_I, K_X, K_Y], n_trials, 20260708)

        # Case 3: same channel, operator list reversed -- frequencies must
        # match the same analytic probabilities regardless of list order
        # (i.e., ordering introduces no bias)
        mixed_rev = KrausGateRep(
            [(K_Y, p_y), (K_X, None), (K_I, 1 - p_x - p_y)], labels
        )
        run_trials(psi, labels, mixed_rev, [K_Y, K_X, K_I], n_trials, 20260708)

        # Case 4: given probabilities summing to slightly below 1 (float
        # roundoff) must not raise -- exercises the tail/renormalization
        # fallback -- and the output state must stay normalized
        eps = 5e-8
        leaky = KrausGateRep(
            [
                (np.sqrt(0.5) * U_X, 0.5),
                (np.sqrt(0.5 - eps) * np.eye(2), 0.5 - eps),
            ],
            labels,
        )
        test = SVState(
            psi.copy(), labels, seed=20260709, kraus_sampling=kraus_sampling
        )
        for _ in range(200):
            test._state = psi.copy()
            test.apply_reps_inplace([leaky])
            assert np.isclose(np.linalg.norm(test.state), 1)

        # Case 5: the same amplitude damping on the first of two qubits,
        # checked against the analytic two-qubit result
        psi_2q = np.kron(psi, np.array([1, 0], dtype=np.complex128))
        A0_2q, A1_2q = np.kron(A0, np.eye(2)), np.kron(A1, np.eye(2))
        amp_damp_2q = KrausGateRep(
            [(A0_2q, None), (A1_2q, None)], ["Q0", "Q1"]
        )
        run_trials(
            psi_2q,
            ["Q0", "Q1"],
            amp_damp_2q,
            [A0_2q, A1_2q],
            n_trials,
            20260710,
        )

    def test_kraus_choice_renormalization_failure(self):
        """`_apply_kraus_choice` must raise a ValueError (not propagate
        numpy's cryptic one) when given probabilities are too far from
        summing to 1 to safely renormalize.

        The deviation must be small enough to pass the method's own
        upfront `np.isclose(sum(probs), 1.0)` sanity assert (~1e-5
        tolerance) but large enough that numpy's `rng.choice` (~1.5e-8
        tolerance) rejects it and larger than the method's own
        renormalization-fallback threshold (1e-7) -- i.e. a deviation of
        about 5e-6."""
        U_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        eps = 2.5e-6
        bad_probs = KrausGateRep(
            [(U_X, 0.5 + eps), (np.eye(2), 0.5 + eps)], ["Q0"]
        )
        test = SVState([0], ["Q0"], seed=1, kraus_sampling="choice")
        with pytest.raises(ValueError):
            test.apply_reps_inplace([bad_probs])

    def test_contraction_matmul_einsum_equivalence(self):
        """The matmul and einsum block-matvec implementations must produce
        the same result for arbitrary (non-unitary, non-Hermitian) operators
        applied to arbitrary qubit subsets, including out-of-order and
        non-adjacent subsets, up to floating-point summation order."""
        n_qubits = 6
        labels = [f"Q{i}" for i in range(n_qubits)]
        rng = np.random.default_rng(20260709)

        # Random dense state (not normalized -- the contraction is linear,
        # so equivalence must hold for any vector)
        vec = rng.standard_normal((2,) * n_qubits) + 1j * rng.standard_normal(
            (2,) * n_qubits
        )

        state_m = SVState(vec.copy(), labels, contraction="matmul")
        state_e = SVState(vec.copy(), labels, contraction="einsum")
        assert state_m.contraction == "matmul"
        assert state_e.contraction == "einsum"

        # 1Q, 2Q, and 3Q targets: in-order, reversed, non-adjacent, endpoints
        subsets = [
            ["Q0"],
            ["Q3"],
            ["Q5"],
            ["Q0", "Q1"],
            ["Q1", "Q0"],
            ["Q3", "Q1"],
            ["Q5", "Q0"],
            ["Q4", "Q2"],
            ["Q1", "Q4", "Q2"],
            ["Q5", "Q0", "Q3"],
        ]
        for sublbls in subsets:
            dim = 2 ** len(sublbls)
            # Random complex operator: no unitarity/symmetry that could
            # mask index-ordering mistakes
            op = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal(
                (dim, dim)
            )
            out_m = state_m._block_matvec(op, sublbls, vec)
            out_e = state_e._block_matvec(op, sublbls, vec)
            assert out_m.shape == out_e.shape == vec.shape
            assert np.allclose(out_m, out_e, atol=1e-13), sublbls

        # Sanity check that the comparison is not trivially symmetric:
        # applying the same operator with swapped target order must NOT
        # match (in either implementation)
        op = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        assert not np.allclose(
            state_m._block_matvec(op, ["Q1", "Q3"], vec),
            state_m._block_matvec(op, ["Q3", "Q1"], vec),
        )
        assert not np.allclose(
            state_e._block_matvec(op, ["Q1", "Q3"], vec),
            state_e._block_matvec(op, ["Q3", "Q1"], vec),
        )

    def test_contraction_equivalence_end_to_end(self):
        """Same seed, same circuit: matmul and einsum contraction modes must
        yield identical measurement/Kraus trajectories and matching states.

        Both modes consume the RNG stream identically (contraction mode does
        not change how randomness is drawn), and their probabilities agree to
        machine precision, so seeded outcomes must match exactly and the
        final statevectors must agree up to summation-order roundoff."""
        n_qubits = 4
        labels = [f"Q{i}" for i in range(n_qubits)]
        seed = 20260709

        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        U_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)

        # 1Q amplitude damping, state-dependent probabilities
        gamma = 0.1
        A0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        A1 = np.array([[0, np.sqrt(gamma)], [0, 0]])

        # 2Q channel with mixed given/None probabilities, applied to an
        # out-of-order, non-adjacent qubit pair
        p_xx = 0.2
        U_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        K_II = np.sqrt(1 - p_xx) * np.eye(4)
        K_XX = np.sqrt(p_xx) * np.kron(U_X, U_X)

        reps = (
            [UnitaryGateRep(U_H, [q]) for q in labels]
            + [
                UnitaryGateRep(U_CZ, ["Q0", "Q1"]),
                UnitaryGateRep(U_CZ, ["Q2", "Q3"]),
            ]
            + [KrausGateRep([(A0, None), (A1, None)], [q]) for q in labels]
            + [
                KrausGateRep([(K_II, 1 - p_xx), (K_XX, None)], ["Q3", "Q1"]),
                ZBasisProjectionInstrumentRep(None, True, ["Q0"]),
                ZBasisProjectionInstrumentRep(0, True, ["Q2"]),
            ]
        )

        def run(contraction):
            state = SVState(
                [0] * n_qubits,
                labels,
                seed=seed,
                kraus_sampling="lazy",
                contraction=contraction,
            )
            outcomes = []
            for _ in range(20):
                outs = state.apply_reps_inplace(reps)
                outcomes.append({k: list(v) for k, v in outs.items()})
            return state, outcomes

        state_m, outcomes_m = run("matmul")
        state_e, outcomes_e = run("einsum")

        assert outcomes_m == outcomes_e
        assert np.allclose(state_m.state, state_e.state, atol=1e-12)

    @pytest.mark.parametrize("include_outcome", [True, False])
    def test_zbasis_outcome_operation_dict_decomposable_multiqubit(
        self, include_outcome
    ):
        """A decomposable multiqubit instrument (`len(outcome_qubits) ==
        len(qubits) > 1`, one classical bit per physical qubit) must select
        exactly the outcome matching the prepared computational basis
        state and record each qubit's own bit, or record nothing (and
        still collapse) when `include_outcome=False`."""

        def basis_projector(bits):
            idx = bits[0] * 2 + bits[1]
            mat = np.zeros((4, 4))
            mat[idx, idx] = 1.0
            return UnitaryGateRep(mat, ["Q0", "Q1"])

        all_bits = [(0, 0), (0, 1), (1, 0), (1, 1)]
        outcome_ops = {bits: basis_projector(bits) for bits in all_bits}
        rep = OutcomeOperationDictInstrumentRep(
            outcome_ops, include_outcome, ["Q0", "Q1"]
        )

        for bits in all_bits:
            test = SVState(list(bits), ["Q0", "Q1"], seed=20260815)
            outs = test.apply_reps_inplace([rep])
            if include_outcome:
                assert outs["Q0"] == [bits[0]]
                assert outs["Q1"] == [bits[1]]
            else:
                assert len(outs) == 0
            expected = np.zeros(4)
            expected[bits[0] * 2 + bits[1]] = 1
            assert np.allclose(np.abs(test.state.flatten()), expected)

    def test_zbasis_outcome_operation_dict_4q_parity_check(self):
        """A 4Q parity-check instrument: the classical outcome reflects
        total computational-basis parity across all 4 qubits, without
        collapsing a superposition confined to one parity sector."""
        qubits = ["Q0", "Q1", "Q2", "Q3"]
        even_diag = np.array(
            [1.0 if bin(i).count("1") % 2 == 0 else 0.0 for i in range(16)]
        )
        odd_diag = 1.0 - even_diag
        even_proj = UnitaryGateRep(np.diag(even_diag), qubits)
        odd_proj = UnitaryGateRep(np.diag(odd_diag), qubits)
        rep = OutcomeOperationDictInstrumentRep(
            {"even": even_proj, "odd": odd_proj},
            True,
            qubits,
            outcome_qubits="synd",
        )

        # |0000> and |1100> both have even parity (popcount 0 and 2).
        state = np.zeros(16)
        state[0b0000] = 1 / np.sqrt(2)
        state[0b1100] = 1 / np.sqrt(2)
        test = SVState(state.copy(), qubits, seed=20260815)
        outs = test.apply_reps_inplace([rep])
        assert outs["synd"] == [0]
        assert np.allclose(test.state.flatten(), state)

    def test_zbasis_outcome_operation_dict_2q_parity_check_samples_both_outcomes(
        self,
    ):
        """This drives a state with support in *both* parity sectors: both
        outcomes must occur with the correct Born-rule frequency, and the
        post-measurement state must exactly match the analytic
        renormalized projection for whichever outcome was sampled."""
        even_proj = UnitaryGateRep(np.diag([1.0, 0, 0, 1.0]), ["Q0", "Q1"])
        odd_proj = UnitaryGateRep(np.diag([0, 1.0, 1.0, 0]), ["Q0", "Q1"])
        rep = OutcomeOperationDictInstrumentRep(
            {"even": even_proj, "odd": odd_proj},
            True,
            ["Q0", "Q1"],
            outcome_qubits="synd_Q0Q1",
        )

        # |00>, |01>, |10>, |11> amplitudes -- unequal so prob_even != 0.5
        # and both sector-internal superpositions are non-trivial.
        psi = np.sqrt([0.5, 0.2, 0.1, 0.2])
        assert np.isclose(np.sum(psi**2), 1)
        prob_even = psi[0] ** 2 + psi[3] ** 2
        even_post = np.array([psi[0], 0, 0, psi[3]]) / np.sqrt(prob_even)
        odd_post = np.array([0, psi[1], psi[2], 0]) / np.sqrt(1 - prob_even)

        test = SVState(psi.copy(), ["Q0", "Q1"], seed=20260815)
        n_trials = 3_000
        counts = [0, 0]
        for _ in range(n_trials):
            test._state = psi.copy().reshape(2, 2)
            outs = test.apply_reps_inplace([rep])
            outcome = outs["synd_Q0Q1"][0]
            counts[outcome] += 1
            expected_post = even_post if outcome == 0 else odd_post
            assert np.allclose(test.state.flatten(), expected_post)

        sigma = np.sqrt(prob_even * (1 - prob_even) / n_trials)
        assert abs(counts[0] / n_trials - prob_even) < 5 * sigma

    def test_zbasis_outcome_operation_dict_joint_channel_with_three_outcomes(
        self,
    ):
        """A joint channel (`len(outcome_qubits) == 1`) has no cardinality
        constraint -- a 3-outcome instrument (e.g. a leakage-aware parity
        check) must sample and record all three outcomes correctly, by
        ordinal position among `outcome_ops`."""
        proj0 = UnitaryGateRep(np.diag([1.0, 0, 0]), ["Q0"], dims=[3])
        proj1 = UnitaryGateRep(np.diag([0, 1.0, 0]), ["Q0"], dims=[3])
        proj2 = UnitaryGateRep(np.diag([0, 0, 1.0]), ["Q0"], dims=[3])
        rep = OutcomeOperationDictInstrumentRep(
            {"ground": proj0, "excited": proj1, "leaked": proj2},
            True,
            ["Q0"],
            outcome_qubits="synd",
        )

        for basis_idx, expected_ordinal in [(0, 0), (1, 1), (2, 2)]:
            state = np.zeros(3)
            state[basis_idx] = 1.0
            test = SVState(state, ["Q0"], seed=20260815, d=3)
            outs = test.apply_reps_inplace([rep])
            assert outs["synd"] == [expected_ordinal]

    def test_zbasis_outcome_operation_dict_lazy_skips_only_after_sampled_branch(
        self,
    ):
        """For a 3-outcome joint channel where the *second* branch is
        dominant, the first branch's matvec must still be computed (its
        probability is needed to know the second branch wasn't yet
        reached), but the third branch's must be skipped entirely."""
        negligible = UnitaryGateRep(np.diag([1e-12, 1e-12]), ["Q0"])
        dominant = UnitaryGateRep(np.eye(2) * np.sqrt(1 - 2e-12), ["Q0"])
        never_applied = UnitaryGateRep(np.eye(2), ["Q0"])
        rep = OutcomeOperationDictInstrumentRep(
            {"a": negligible, "b": dominant, "c": never_applied},
            True,
            ["Q0"],
            outcome_qubits="synd",
        )
        test = SVState([0], ["Q0"], seed=20260815)

        call_count = 0
        original_matvec = test._block_matvec

        def counting_matvec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_matvec(*args, **kwargs)

        test._block_matvec = counting_matvec
        outs = test.apply_reps_inplace([rep])
        assert outs["synd"] == [1]  # "b" (dominant) is ordinal 1
        assert call_count == 2  # "a" computed, "b" computed, "c" skipped

    def test_qutrit_init(self):
        # 1. Base qutrit initializer
        qubit_labels = ["Q0", "Q1"]
        s = SVState(2, qubit_labels, d=3)
        assert s.state.shape == (3, 3)
        assert s.d == [3, 3]

        # Copy constructor
        s2 = SVState(s)
        self._check(s2, s)

        # Bitstring initializer (e.g., state |012> on 3 qutrits)
        qubit_labels_3 = ["Q0", "Q1", "Q2"]
        s3 = SVState([0, 1, 2], qubit_labels_3, d=3)
        assert s3.state.shape == (3, 3, 3)
        assert np.allclose(s3.state[0, 1, 2], 1.0)

        # Cast with flat numpy array check
        flat_arr = np.zeros(27, dtype=complex)
        flat_arr[5] = (
            1.0  # corresponds to index [0, 1, 2] since 0*9 + 1*3 + 2 = 5
        )
        s4 = SVState(flat_arr, qubit_labels_3, d=3)
        assert s4.d == [3, 3, 3]
        assert s4.state.shape == (3, 3, 3)
        assert np.allclose(s4.state[0, 1, 2], 1.0)

        # A flat array with no qubit labels infers the number of subsystems
        # with a base-d logarithm; d=3 must not be mistaken for base-2.
        flat_arr = np.zeros(27, dtype=complex)
        flat_arr[0] = 1.0
        s5 = SVState(flat_arr, d=3)
        assert len(s5.qubit_labels) == 3
        assert s5.d == [3, 3, 3]
        assert s5.state.shape == (3, 3, 3)

    def test_qutrit_projective_measurement(self):
        # Equal superposition of all 3 levels: (|0> + |1> + |2>) / sqrt(3)
        superposition = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
        state = SVState(superposition, ["Q0"], d=3, seed=12345)

        # Perform 300 measurements with reset=None (leaves them in the projected state)
        outcomes = []
        iz_rep = [ZBasisProjectionInstrumentRep(None, True, ["Q0"])]
        for _ in range(300):
            test = state.copy()
            test._rng = state._rng  # Share advancing RNG
            res = test.apply_reps_inplace(iz_rep)
            outcomes.append(res["Q0"][0])

        # Verify that all three outcomes occur roughly with ~1/3 probability
        counts = np.bincount(outcomes, minlength=3)
        assert all(c > 60 for c in counts)

    def test_mixed_qubit_qutrit(self, make_temp_path):
        # 1. Initialize mixed state: Q0 is qubit (d=2), Q1 is qutrit (d=3)
        qubit_labels = ["Q0", "Q1"]
        state = SVState([0, 0], qubit_labels, d=[2, 3])
        assert state.state.shape == (2, 3)
        assert state.d == [2, 3]

        # 2. Apply qubit-only unitary gate (2x2) on Q0
        U_X_2 = np.array([[0, 1], [1, 0]], dtype=complex)
        state.apply_reps_inplace([UnitaryGateRep(U_X_2, ["Q0"])])
        # Expected state is |10>
        expected1 = SVState([1, 0], qubit_labels, d=[2, 3])
        assert np.allclose(state.state, expected1.state)

        # 3. Apply qutrit-only unitary gate (3x3) on Q1 (swap 0 <-> 1)
        U_X_3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
        state.apply_reps_inplace([UnitaryGateRep(U_X_3, ["Q1"], dims=[3])])
        # Expected state is |11>
        expected2 = SVState([1, 1], qubit_labels, d=[2, 3])
        assert np.allclose(state.state, expected2.state)

        # 4. Apply joint qubit-qutrit unitary gate (6x6) on Q0, Q1
        # Swaps index 4 (|11>) and index 5 (|12>)
        U_joint = np.eye(6, dtype=complex)
        U_joint[4, 4] = 0.0
        U_joint[5, 5] = 0.0
        U_joint[4, 5] = 1.0
        U_joint[5, 4] = 1.0

        state.apply_reps_inplace(
            [UnitaryGateRep(U_joint, ["Q0", "Q1"], dims=[2, 3])]
        )
        # Expected state is |12>
        expected3 = SVState([1, 2], qubit_labels, d=[2, 3])
        assert np.allclose(state.state, expected3.state)

        # 5. Measure qubit in Z-basis with reset=0
        iz_reset_rep = [ZBasisProjectionInstrumentRep(0, True, ["Q0"])]
        res = state.apply_reps_inplace(iz_reset_rep)
        assert res["Q0"][0] == 1  # was prepared as |1>
        # Q0 is now reset to |0>, so state is |02>
        expected4 = SVState([0, 2], qubit_labels, d=[2, 3])
        assert np.allclose(state.state, expected4.state)

        # 6. Mixed state serialization/deserialization
        with make_temp_path(suffix=".json") as tmp_path:
            state.write(tmp_path)
            loaded = SVState.read(tmp_path)
        self._check(loaded, state)

    def test_serialization_kraus_and_instruments(self, make_temp_path):
        """Serialization must round-trip non-default
        `kraus_sampling`/`contraction` settings (both `_SERIALIZE_ATTRS`)
        along with the state."""
        U_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        X_kraus = KrausGateRep([(U_X, 1.0), (np.zeros((2, 2)), 0.0)], ["Q0"])
        proj_rep = ZBasisProjectionInstrumentRep(None, True, ["Q0"])

        test = SVState(
            [0],
            ["Q0"],
            seed=20260711,
            kraus_sampling="choice",
            contraction="einsum",
        )
        outs_before = test.apply_reps_inplace([X_kraus, proj_rep])

        with make_temp_path(suffix=".json") as tmp_path:
            test.write(tmp_path)
            test2 = SVState.read(tmp_path)

        assert isinstance(test2, SVState)
        assert test2.kraus_sampling == "choice"
        assert test2.contraction == "einsum"
        self._check(test2, test)
        # Deterministic Kraus channel (prob 1.0/0.0) always flips to |1>,
        # so the projector must always measure 1 both before and after
        # the round trip.
        assert outs_before["Q0"] == [1]
