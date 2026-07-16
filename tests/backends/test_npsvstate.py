"""Tester for loqs.backends.state.qsimstate"""

import os
import tempfile
import json

import mock
import numpy as np
import pytest

from loqs.backends.reps import GateRep, RepTuple, InstrumentRep
from loqs.backends import NumpyStatevectorQuantumState as SVState


class TestNumPyStatevectorQuantumState:

    def _check(self, state, expected_state):
        assert state.qubit_labels == expected_state.qubit_labels
        assert state.seed == expected_state.seed
        assert np.allclose(state.state, expected_state.state)

    def test_init(self):
        # Base initializer
        qubit_labels = [f"Q{i}" for i in range(5)]
        s = SVState(5, qubit_labels)

        s2 = SVState(s)
        self._check(s2, s)

        # Bitstring initializer
        s3 = SVState([0, 0, 0, 0, 0], qubit_labels)
        self._check(s3, s)

        # Cast checks
        s4 = SVState.cast(s)
        self._check(s4, s)

        # (Flat) Numpy array check also
        all0_state = np.zeros(2**5)
        all0_state[0] = 1
        s5 = SVState.cast((all0_state, qubit_labels))
        self._check(s5, s)

        # This one won't have same labels
        s6 = SVState.cast(5)
        self._check(SVState(5), s6)

        # Copy check
        s7 = s.copy()
        self._check(s7, s)
    
    def test_apply_gates(self):
        # Let's apply a X gate
        U_X = np.array([[0, 1], [1, 0]])
        X_reps = [RepTuple(U_X, ["Q0"], GateRep.UNITARY)]

        # Start in the 0 state
        state0 = SVState([0], ["Q0"])

        # Also prepare a 1 state as expected
        state1 = SVState([1], ["Q0"])

        # Test both in-place and not
        test = state0.copy()
        test.apply_reps_inplace(X_reps)
        self._check(test, state1)
        
        test2, outcomes = state0.apply_reps(X_reps)
        self._check(test2, state1)
        assert len(outcomes) == 0

        # Let's try a CNOT via H CZ H
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        U_CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        CX_reps = [
            RepTuple(U_H, ["Q1"], GateRep.UNITARY),
            RepTuple(U_CZ, ["Q0", "Q1"], GateRep.UNITARY),
            RepTuple(U_H, ["Q1"], GateRep.UNITARY)
        ]

        # Start in the |10> (big-endian) state
        state10 = SVState([1, 0], ["Q0", "Q1"])
        
        # The expected |11> state
        state11 = SVState([1, 1], ["Q0", "Q1"])

        test3, _ = state10.apply_reps(CX_reps)
        self._check(test3, state11)

        # TODO: Test Kraus
        # Test Kraus operator where applying X with prob 1, and I with prob 0
        X_kraus_rep_w_prob = RepTuple([(U_X, 1.0), (np.eye(2), 0.0)], ["Q0"], GateRep.KRAUS_OPERATORS)
        for _ in range(10):
            test4 = state0.copy()
            test4.apply_reps_inplace([X_kraus_rep_w_prob])
            self._check(test4, state1)
        
        # Test Kraus operator where bitflip happens with half the time
        outcomes1 = []
        half_bitflip_w_prob = RepTuple([(1/np.sqrt(2)*U_X, 0.5), (1/np.sqrt(2)*np.eye(2), 0.5)], ["Q0"], GateRep.KRAUS_OPERATORS)
        test5 = SVState([0], ["Q0"], seed=20260122)
        for _ in range(10):
            test5.apply_reps_inplace([half_bitflip_w_prob])
            # Store if we are in 1 state or 0 state
            outcomes1.append(test5.state[1])
            # Manual reset
            test5._state[0] = 1
            test5._state[1] = 0
        # outcomes should not be all 0 or 1
        assert any([np.isclose(o, 0) for o in outcomes1]) and any([np.isclose(o, 1) for o in outcomes1])

        # Lets do the same half bitflip, but force probability computation
        # Outcomes should be the same if we seed the same
        outcomes2 = []
        half_bitflip_wout_prob = RepTuple([(1/np.sqrt(2)*U_X, None), (1/np.sqrt(2)*np.eye(2), None)], ["Q0"], GateRep.KRAUS_OPERATORS)
        test6 = SVState([0], ["Q0"], seed=20260122)
        for _ in range(10):
            test6.apply_reps_inplace([half_bitflip_wout_prob])
            # Store if we are in 1 state or 0 state
            outcomes2.append(test6.state[1])
            # Manual reset
            test6._state[0] = 1
            test6._state[1] = 0
        # Outcomes should be the same since we seeded the same
        assert np.allclose(outcomes1, outcomes2)

        # Let's try to pass in some unsupported reps
        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.PTM)
            ])
        
        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.STIM_CIRCUIT_STR)
            ])

        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.QSIM_SUPEROPERATOR)
            ])

        with pytest.raises(ValueError):
            test.apply_reps([
                RepTuple(None, "Q0", GateRep.PROBABILISTIC_STIM_OPERATIONS)
            ])

    def test_input_reps(self):
        state = SVState(1, ["Q0"])
        assert set(state.input_reps) == {
            GateRep.UNITARY,
            GateRep.KRAUS_OPERATORS,
            InstrumentRep.ZBASIS_PROJECTION,
            InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
            InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
        }

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

        def run_trials(reptuple, Ks, n_trials, seed):
            # Analytic branch probabilities and post-application states
            probs_exact = [np.linalg.norm(K @ psi) ** 2 for K in Ks]
            posts_exact = [K @ psi / np.linalg.norm(K @ psi) for K in Ks]
            assert np.isclose(sum(probs_exact), 1)

            test = SVState(
                psi.copy(), ["Q0"], seed=seed, kraus_sampling=kraus_sampling
            )
            counts = [0] * len(Ks)
            for _ in range(n_trials):
                test._state = psi.copy()  # manual reset, RNG stream continues
                test.apply_reps_inplace([reptuple])

                # Identify the branch by phase-insensitive overlap; the output
                # must *exactly* match that branch's normalized K_i |psi>
                fids = [
                    np.abs(np.vdot(post, test.state)) for post in posts_exact
                ]
                branch = int(np.argmax(fids))
                assert np.isclose(fids[branch], 1)
                counts[branch] += 1

            # Frequencies within 5 sigma (binomial) of Born probabilities --
            # deterministic given the seed, so no flakiness
            for count, p in zip(counts, probs_exact):
                sigma = np.sqrt(p * (1 - p) / n_trials)
                assert abs(count / n_trials - p) < 5 * sigma

        # Case 1: amplitude damping with all probabilities None
        # (exercises the state-dependent probability path)
        gamma = 0.3
        A0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        A1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        amp_damp = RepTuple(
            [(A0, None), (A1, None)], ["Q0"], GateRep.KRAUS_OPERATORS
        )
        run_trials(amp_damp, [A0, A1], n_trials=20_000, seed=20260708)

        # Case 2: three operators with a mix of given and None probabilities
        # (exercises mixed accumulation across the operator list)
        p_x, p_y = 0.2, 0.1
        U_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        U_Y = np.array([[0, -1j], [1j, 0]])
        K_I = np.sqrt(1 - p_x - p_y) * np.eye(2)
        K_X = np.sqrt(p_x) * U_X
        K_Y = np.sqrt(p_y) * U_Y
        mixed = RepTuple(
            [(K_I, 1 - p_x - p_y), (K_X, None), (K_Y, p_y)],
            ["Q0"],
            GateRep.KRAUS_OPERATORS,
        )
        run_trials(mixed, [K_I, K_X, K_Y], n_trials=20_000, seed=20260708)

        # Case 3: same channel, operator list reversed -- frequencies must
        # match the same analytic probabilities regardless of list order
        # (i.e., ordering introduces no bias)
        mixed_rev = RepTuple(
            [(K_Y, p_y), (K_X, None), (K_I, 1 - p_x - p_y)],
            ["Q0"],
            GateRep.KRAUS_OPERATORS,
        )
        run_trials(mixed_rev, [K_Y, K_X, K_I], n_trials=20_000, seed=20260708)

        # Case 4: given probabilities summing to slightly below 1 (float
        # roundoff) must not raise -- exercises the tail/renormalization
        # fallback -- and the output state must stay normalized
        eps = 5e-8
        leaky = RepTuple(
            [
                (np.sqrt(0.5) * U_X, 0.5),
                (np.sqrt(0.5 - eps) * np.eye(2), 0.5 - eps),
            ],
            ["Q0"],
            GateRep.KRAUS_OPERATORS,
        )
        test = SVState(
            psi.copy(), ["Q0"], seed=20260709, kraus_sampling=kraus_sampling
        )
        for _ in range(200):
            test._state = psi.copy()
            test.apply_reps_inplace([leaky])
            assert np.isclose(np.linalg.norm(test.state), 1)

    def test_kraus_choice_renormalization_failure(self):
        """`_apply_kraus_choice` must raise a clear ValueError (not
        propagate numpy's cryptic one) when given probabilities are too
        far from summing to 1 to safely renormalize.

        The deviation must be small enough to pass the method's own
        upfront `np.isclose(sum(probs), 1.0)` sanity assert (~1e-5
        tolerance) but large enough that numpy's `rng.choice` (~1.5e-8
        tolerance) rejects it and larger than the method's own
        renormalization-fallback threshold (1e-7) -- i.e. a deviation of
        about 5e-6."""
        U_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        eps = 2.5e-6
        bad_probs = RepTuple(
            [(U_X, 0.5 + eps), (np.eye(2), 0.5 + eps)],
            ["Q0"],
            GateRep.KRAUS_OPERATORS,
        )
        test = SVState([0], ["Q0"], seed=1, kraus_sampling="choice")
        with pytest.raises(ValueError, match="too far from 1 to renormalize"):
            test.apply_reps_inplace([bad_probs])

    @pytest.mark.parametrize("kraus_sampling", ["lazy", "choice"])
    def test_kraus_sampling_distribution_multiqubit(self, kraus_sampling):
        """Same Born-rule/exact-post-state contract as
        `test_kraus_sampling_distribution`, but for 2- and 3-qubit Kraus
        operators, cross-checked against an independent analytic ground
        truth (not just internal matmul-vs-einsum agreement)."""

        def run_trials(psi, labels, Ks, n_trials, seed):
            n_qubits = len(labels)
            probs_exact = [np.linalg.norm(K @ psi) ** 2 for K in Ks]
            posts_exact = [K @ psi / np.linalg.norm(K @ psi) for K in Ks]
            assert np.isclose(sum(probs_exact), 1)

            reptuple = RepTuple(
                [(K, None) for K in Ks], labels, GateRep.KRAUS_OPERATORS
            )
            test = SVState(
                psi.copy().reshape((2,) * n_qubits),
                labels,
                seed=seed,
                kraus_sampling=kraus_sampling,
            )
            counts = [0] * len(Ks)
            for _ in range(n_trials):
                test._state = psi.copy().reshape((2,) * n_qubits)
                test.apply_reps_inplace([reptuple])
                out_flat = test.state.flatten()
                fids = [
                    np.abs(np.vdot(post, out_flat)) for post in posts_exact
                ]
                branch = int(np.argmax(fids))
                assert np.isclose(fids[branch], 1)
                counts[branch] += 1

            for count, p in zip(counts, probs_exact):
                sigma = np.sqrt(p * (1 - p) / n_trials)
                assert abs(count / n_trials - p) < 5 * sigma

        # Amplitude damping on the first qubit, identity elsewhere --
        # kron'd up to 2 and 3 qubits so probabilities stay state-dependent
        # (unlike e.g. two unitary branches, whose branch probabilities
        # would be state-independent and so a much weaker test).
        gamma = 0.3
        A0_1q = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        A1_1q = np.array([[0, np.sqrt(gamma)], [0, 0]])

        theta = 0.4
        psi_1q = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
        other_q = np.array([1, 0], dtype=np.complex128)

        psi_2q = np.kron(psi_1q, other_q)
        A0_2q, A1_2q = np.kron(A0_1q, np.eye(2)), np.kron(A1_1q, np.eye(2))
        run_trials(
            psi_2q, ["Q0", "Q1"], [A0_2q, A1_2q], n_trials=20_000, seed=20260710
        )

        psi_3q = np.kron(psi_2q, other_q)
        A0_3q, A1_3q = np.kron(A0_2q, np.eye(2)), np.kron(A1_2q, np.eye(2))
        run_trials(
            psi_3q,
            ["Q0", "Q1", "Q2"],
            [A0_3q, A1_3q],
            n_trials=10_000,
            seed=20260710,
        )

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
        vec = rng.standard_normal(
            (2,) * n_qubits
        ) + 1j * rng.standard_normal((2,) * n_qubits)

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

    @pytest.mark.parametrize("kraus_sampling", ["lazy", "choice"])
    def test_contraction_equivalence_end_to_end(self, kraus_sampling):
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
            [RepTuple(U_H, [q], GateRep.UNITARY) for q in labels]
            + [
                RepTuple(U_CZ, ["Q0", "Q1"], GateRep.UNITARY),
                RepTuple(U_CZ, ["Q2", "Q3"], GateRep.UNITARY),
            ]
            + [
                RepTuple(
                    [(A0, None), (A1, None)], [q], GateRep.KRAUS_OPERATORS
                )
                for q in labels
            ]
            + [
                RepTuple(
                    [(K_II, 1 - p_xx), (K_XX, None)],
                    ["Q3", "Q1"],
                    GateRep.KRAUS_OPERATORS,
                ),
                RepTuple(
                    (None, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION
                ),
                RepTuple((0, True), ["Q2"], InstrumentRep.ZBASIS_PROJECTION),
            ]
        )

        def run(contraction):
            state = SVState(
                [0] * n_qubits,
                labels,
                seed=seed,
                kraus_sampling=kraus_sampling,
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

    def test_apply_instruments(self):
        # H gate to get + state for testing
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_rep = RepTuple(U_H, ["Q0"], GateRep.UNITARY)

        state0 = SVState([0], ["Q0"], seed=20241016)

        state1 = SVState([1], ["Q0"], seed=20241016)

        # In-place 10 times
        # Also test no reset
        proj_rep = RepTuple((None, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test = state0.copy()
        outcomes1 = []
        for _ in range(10):
            outs = test.apply_reps_inplace([H_rep, proj_rep])
            out = outs["Q0"][0]
            outcomes1.append(out)

            # Check measurement without reset
            if out == 1:
                self._check(test, state1)
                # Reset manually
                test.state[1] = 0
                test.state[0] = 1
            else:
                self._check(test, state0)
        
        # Also test no outcomes
        proj2_rep = RepTuple((None, False), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test1 = state0.copy()
        outs = test1.apply_reps_inplace([H_rep, proj2_rep]*10)
        assert len(outs) == 0
        
        # Now another copy ten times at once with reset
        reset_rep = RepTuple((0, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)
        test2 = state0.copy()
        outs = test2.apply_reps_inplace([H_rep, reset_rep]*10)
        outcomes2 = outs["Q0"]
        
        # Should be same outcomes because of RNG seeding
        assert outcomes1 == outcomes2

        # Now lets test pre/post op
        U_I = np.eye(2)
        idle_rep = RepTuple(U_I, ["Q0"], GateRep.UNITARY)

        # Lets do X(pi/2) error before and nothing after
        pre_H_rep = RepTuple(
            [0, True, H_rep, idle_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test3 = state0.copy()
        outs = test3.apply_reps_inplace([pre_H_rep]*10)
        outcomes3 = outs["Q0"]
        assert outcomes3 == outcomes1

        # Now let's do X(pi/2) after and no nothing before
        # Very first one we have to do X(pi/2) to get same outcomes
        post_H_rep = RepTuple(
            [0, True, idle_rep, H_rep], ["Q0"], InstrumentRep.ZBASIS_PRE_POST_OPERATIONS
        )

        test4 = state0.copy()
        outs = test4.apply_reps_inplace([H_rep] + [post_H_rep]*10)
        outcomes4 = outs["Q0"]
        assert outcomes4 == outcomes1

        # Finally let's do the outcome/operation dict
        effect0 = np.array([[1, 0]])
        effect1 = np.array([[0, 1]])

        ideal_maps = {
            0: RepTuple(effect0.T @ effect0, ["Q0"], GateRep.UNITARY),
            1: RepTuple(effect1.T @ effect1, ["Q0"], GateRep.UNITARY)
        }
        ideal_map_rep = RepTuple((ideal_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test5 = state0.copy()
        outs = test5.apply_reps_inplace([H_rep, ideal_map_rep]*10)
        outcomes5 = outs["Q0"]
        assert outcomes5 == outcomes1

        # Let's use the instrument to also do reset
        reset_maps = {
            0: RepTuple(effect0.T @ effect0, ["Q0"], GateRep.UNITARY),
            1: RepTuple(effect0.T @ effect1, ["Q0"], GateRep.UNITARY)
        }
        reset_map_rep = RepTuple((reset_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test6 = state0.copy()
        outs = test6.apply_reps_inplace([H_rep, reset_map_rep]*10)
        outcomes6 = outs["Q0"]
        assert outcomes6 == outcomes1

        noisy_reset_maps = {
            0: RepTuple(U_H @ effect0.T @ effect0, ["Q0"], GateRep.UNITARY),
            1: RepTuple(U_H @ effect0.T @ effect1, ["Q0"], GateRep.UNITARY)
        }
        noisy_reset_map_rep = RepTuple((noisy_reset_maps, True), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT)

        test7 = state0.copy()
        outs = test7.apply_reps_inplace([H_rep] + [noisy_reset_map_rep]*10)
        outcomes7 = outs["Q0"]
        assert outcomes7 == outcomes1

    def test_unsupported_instrument_rep_raises(self):
        test = SVState([0], ["Q0"])
        with pytest.raises(NotImplementedError):
            test.apply_reps_inplace([
                RepTuple("M 0", ["Q0"], InstrumentRep.STIM_CIRCUIT_STR)
            ])

    def test_zbasis_projection_reset_to_1(self):
        """`reset=1` must always leave the qubit in |1> (up to the
        physically irrelevant global phase carried over from whichever
        branch was measured -- each trial starts fresh from |0> to avoid
        that phase compounding across reused, un-reset iterations)."""
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_rep = RepTuple(U_H, ["Q0"], GateRep.UNITARY)
        reset1_rep = RepTuple((1, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION)

        for trial in range(10):
            test = SVState([0], ["Q0"], seed=20260711 + trial)
            test.apply_reps_inplace([H_rep, reset1_rep])
            assert np.allclose(np.abs(test.state), [0, 1])

    def test_zbasis_projection_multiqubit(self):
        """A single ZBASIS_PROJECTION rep applied to multiple qubits at
        once must measure/reset every qubit in `qubits`."""
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_reps = [
            RepTuple(U_H, ["Q0"], GateRep.UNITARY),
            RepTuple(U_H, ["Q1"], GateRep.UNITARY),
        ]
        reset0_rep = RepTuple(
            (0, True), ["Q0", "Q1"], InstrumentRep.ZBASIS_PROJECTION
        )

        state00 = SVState([0, 0], ["Q0", "Q1"], seed=20260711)
        test = SVState([0, 0], ["Q0", "Q1"], seed=20260711)
        for _ in range(10):
            outs = test.apply_reps_inplace(H_reps + [reset0_rep])
            assert set(outs.keys()) == {"Q0", "Q1"}
            assert len(outs["Q0"]) == 1 and len(outs["Q1"]) == 1
            # reset=0 must always leave both qubits in |00>
            self._check(test, state00)

    def test_zbasis_pre_post_operations_reset_and_no_outcomes(self):
        """ZBASIS_PRE_POST_OPERATIONS with `reset=1` and
        `include_outcomes=False` must suppress the outcome dict entry
        while still always leaving the qubit reset to |1>."""
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_rep = RepTuple(U_H, ["Q0"], GateRep.UNITARY)
        U_I = np.eye(2)
        idle_rep = RepTuple(U_I, ["Q0"], GateRep.UNITARY)

        # reset=1, include_outcomes=False: measurement outcome should not
        # be recorded, but the qubit must still always end up in |1> (up
        # to global phase -- fresh state per trial, as above)
        pre_H_reset1_no_outcomes = RepTuple(
            [1, False, H_rep, idle_rep],
            ["Q0"],
            InstrumentRep.ZBASIS_PRE_POST_OPERATIONS,
        )
        for trial in range(10):
            test = SVState([0], ["Q0"], seed=20260711 + trial)
            outs = test.apply_reps_inplace([pre_H_reset1_no_outcomes])
            assert len(outs) == 0
            assert np.allclose(np.abs(test.state), [0, 1])

    def test_zbasis_outcome_operation_dict_multiqubit_raises(self):
        """ZBASIS_OUTCOME_OPERATION_DICT explicitly does not support more
        than one qubit."""
        dummy_maps = {0: object(), 1: object()}
        rep = RepTuple(
            (dummy_maps, True),
            ["Q0", "Q1"],
            InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT,
        )
        test = SVState([0, 0], ["Q0", "Q1"])
        with pytest.raises(NotImplementedError):
            test.apply_reps_inplace([rep])

    def test_zbasis_outcome_operation_dict_no_outcomes_and_final_state(self):
        """`include_outcomes=False` for ZBASIS_OUTCOME_OPERATION_DICT must
        suppress the outcome dict entry, while the ideal projector map
        must still always collapse to an exact computational basis
        state."""
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        H_rep = RepTuple(U_H, ["Q0"], GateRep.UNITARY)

        effect0 = np.array([[1, 0]])
        effect1 = np.array([[0, 1]])
        ideal_maps = {
            0: RepTuple(effect0.T @ effect0, ["Q0"], GateRep.UNITARY),
            1: RepTuple(effect1.T @ effect1, ["Q0"], GateRep.UNITARY),
        }
        ideal_map_rep_no_outcomes = RepTuple(
            (ideal_maps, False), ["Q0"], InstrumentRep.ZBASIS_OUTCOME_OPERATION_DICT
        )

        for trial in range(10):
            test = SVState([0], ["Q0"], seed=20260711 + trial)
            outs = test.apply_reps_inplace([H_rep, ideal_map_rep_no_outcomes])
            assert len(outs) == 0
            # The ideal projector map collapses to the basis state matching
            # whichever outcome was (invisibly) sampled -- so the final
            # state must always be an exact computational basis state
            # (up to global phase).
            assert np.allclose(np.abs(test.state), [1, 0]) or np.allclose(
                np.abs(test.state), [0, 1]
            )

    def test_serialization(self, make_temp_path):
        # Start in the 10 state
        state10 = SVState([1, 0], ["Q0", "Q1"])
        
        # Let's try a CNOT via H CZ H
        # But let's split the H CZ before serialization
        # and final H after serialization
        U_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        U_CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        test, _ = state10.apply_reps([RepTuple(U_H, ["Q1"], GateRep.UNITARY)])
        test.apply_reps_inplace([RepTuple(U_CZ, ["Q0", "Q1"], GateRep.UNITARY)])

        with make_temp_path(suffix='.json') as tmp_path:
            test.write(tmp_path)
            test2 = SVState.read(tmp_path)
        
        # And finish applying
        assert isinstance(test2, SVState)
        test2.apply_reps_inplace([RepTuple(U_H, ["Q1"], GateRep.UNITARY)])
        
        # The expected 11 state
        state11 = SVState([1, 1], ["Q0", "Q1"])

        self._check(test2, state11)

    @pytest.mark.parametrize("kraus_sampling", ["lazy", "choice"])
    @pytest.mark.parametrize("contraction", ["matmul", "einsum"])
    def test_serialization_kraus_and_instruments(
        self, make_temp_path, kraus_sampling, contraction
    ):
        """Serialization must round-trip KRAUS_OPERATORS gate application
        and an InstrumentRep correctly, along with non-default
        `kraus_sampling`/`contraction` settings (both `_SERIALIZE_ATTRS`)."""
        U_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        X_kraus = RepTuple(
            [(U_X, 1.0), (np.eye(2), 0.0)], ["Q0"], GateRep.KRAUS_OPERATORS
        )
        proj_rep = RepTuple(
            (None, True), ["Q0"], InstrumentRep.ZBASIS_PROJECTION
        )

        test = SVState(
            [0], ["Q0"], seed=20260711,
            kraus_sampling=kraus_sampling, contraction=contraction,
        )
        outs_before = test.apply_reps_inplace([X_kraus, proj_rep])

        with make_temp_path(suffix=".json") as tmp_path:
            test.write(tmp_path)
            test2 = SVState.read(tmp_path)

        assert isinstance(test2, SVState)
        assert test2.kraus_sampling == kraus_sampling
        assert test2.contraction == contraction
        self._check(test2, test)
        # Deterministic Kraus channel (prob 1.0/0.0) always flips to |1>,
        # so the projector must always measure 1 both before and after
        # the round trip.
        assert outs_before["Q0"] == [1]

