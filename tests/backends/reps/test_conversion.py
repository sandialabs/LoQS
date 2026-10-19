"""Tester for loqs.backends.reps.conversion"""

import numpy as np
import pytest

from loqs.backends.reps import (
    KrausGateRep,
    PTMGateRep,
    QSimSuperopGateRep,
    RepConstructionError,
    STANDARD_GATE_UNITARIES,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    OutcomeOperationDictInstrumentRep,
    ZBasisPrePostInstrumentRep,
    ZBasisProjectionInstrumentRep,
)
from loqs.backends.reps.conversion import (
    _is_identity_gaterep,
    _outcome_operation_dict_to_zbasis_projection,
    _ptm_to_kraus,
    _ptm_to_qsim_superoperator,
    _ptm_to_unitary,
    _kraus_to_ptm,
    _kraus_to_unitary,
    _qsim_basis,
    _qsim_superoperator_to_ptm,
    _shortest_path,
    _stim_circuit_to_unitary,
    _stim_circuit_to_zbasis_projection,
    _unitary_to_kraus,
    _unitary_to_ptm,
    _unitary_to_stim_circuit,
    _zbasis_pre_post_to_zbasis_projection,
    _zbasis_projection_to_outcome_operation_dict,
    _zbasis_projection_to_stim_circuit,
    _zbasis_projection_to_zbasis_pre_post,
)

try:
    import pygsti

    NO_PYGSTI = False
except ImportError:
    NO_PYGSTI = True

try:
    import stim  # noqa: F401 -- import itself is the skip-guard for missing stim

    NO_STIM = False
except ImportError:
    NO_STIM = True


def _depolarizing_kraus_ops(p: float) -> list[tuple[np.ndarray, None]]:
    X, Y, Z = (
        STANDARD_GATE_UNITARIES["X"],
        STANDARD_GATE_UNITARIES["Y"],
        STANDARD_GATE_UNITARIES["Z"],
    )
    return [
        (np.sqrt(1 - 3 * p / 4) * np.eye(2), None),
        (np.sqrt(p / 4) * X, None),
        (np.sqrt(p / 4) * Y, None),
        (np.sqrt(p / 4) * Z, None),
    ]


def _amplitude_damping_kraus_ops(
    gamma: float,
) -> list[tuple[np.ndarray, None]]:
    A0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    A1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return [(A0, None), (A1, None)]


def _channel_action(kraus_ops, rho: np.ndarray) -> np.ndarray:
    return sum(K @ rho @ K.conj().T for K, _ in kraus_ops)


def _random_density_matrices(d: int, n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        yield A @ A.conj().T


def _assert_same_channel_action(kraus_ops1, kraus_ops2, d: int):
    for rho in _random_density_matrices(d):
        out1 = _channel_action(kraus_ops1, rho)
        out2 = _channel_action(kraus_ops2, rho)
        assert np.allclose(out1, out2, atol=1e-7)


class TestUnitaryToPTM:
    @pytest.mark.skipif(NO_PYGSTI, reason="pyGSTi not installed")
    @pytest.mark.parametrize("name", ["H", "S", "CX"])
    def test_matches_pygsti_unitary_to_pauligate(self, name):
        U = STANDARD_GATE_UNITARIES[name]
        qubits = ("Q0", "Q1") if U.shape[0] == 4 else ("Q0",)
        rep = UnitaryGateRep(U, qubits)
        mine = _unitary_to_ptm(rep).ptm
        theirs = np.asarray(pygsti.tools.unitary_to_pauligate(U))
        assert np.allclose(mine, theirs, atol=1e-8)


class TestKrausToPTM:
    def test_matches_unitary_to_ptm_for_single_kraus_term(self):
        H = STANDARD_GATE_UNITARIES["H"]
        unitary_ptm = _unitary_to_ptm(UnitaryGateRep(H, ("Q0",))).ptm
        kraus_ptm = _kraus_to_ptm(KrausGateRep([(H, 1.0)], ("Q0",))).ptm
        assert np.allclose(unitary_ptm, kraus_ptm)


class TestUnitaryToKraus:
    def test_trivial_single_operator(self):
        H = STANDARD_GATE_UNITARIES["H"]
        rep = _unitary_to_kraus(UnitaryGateRep(H, ("Q0",)))
        assert isinstance(rep, KrausGateRep)
        assert len(rep.kraus_operators) == 1
        K, prob = rep.kraus_operators[0]
        assert np.array_equal(K, H)
        assert prob == 1.0


class TestPTMToKraus:
    @pytest.mark.parametrize("channel", ["unitary", "amplitude-damping"])
    def test_channel_action_preserved(self, channel):
        if channel == "unitary":
            CX = STANDARD_GATE_UNITARIES["CX"]
            true_ops, d = [(CX, None)], 4
            ptm_rep = _unitary_to_ptm(UnitaryGateRep(CX, ("Q0", "Q1")))
        else:
            # The Choi decomposition's reshape needs a transpose to be
            # correct for asymmetric operators like amplitude damping's.
            true_ops, d = _amplitude_damping_kraus_ops(0.4), 2
            ptm_rep = _kraus_to_ptm(KrausGateRep(true_ops, ("Q0",)))
        recovered = _ptm_to_kraus(ptm_rep)
        assert len(recovered.kraus_operators) == len(true_ops)
        _assert_same_channel_action(recovered.kraus_operators, true_ops, d=d)
        if channel == "amplitude-damping":
            # Not proportional to a unitary, so probabilities are
            # state-dependent and stay unset.
            assert all(prob is None for _, prob in recovered.kraus_operators)

    def test_precomputes_fixed_probability_for_scaled_unitary_terms(self):
        """A probabilistic-bit-flip channel's Kraus terms are each
        proportional to a unitary (once their own scale is divided out),
        so each should get a precomputed, non-`None` probability -- not
        left for state backends to compute at simulation time."""
        X = STANDARD_GATE_UNITARIES["X"]
        p = 0.2
        true_ops = [
            (np.sqrt(1 - p) * np.eye(2), None),
            (np.sqrt(p) * X, None),
        ]
        ptm_rep = _kraus_to_ptm(KrausGateRep(true_ops, ("Q0",)))
        recovered = _ptm_to_kraus(ptm_rep)
        probs = sorted(prob for _, prob in recovered.kraus_operators)
        assert all(prob is not None for prob in probs)
        assert np.allclose(probs, sorted([p, 1 - p]))


class TestPTMToUnitary:
    def test_succeeds_for_unitary_channel(self):
        H = STANDARD_GATE_UNITARIES["H"]
        ptm_rep = _unitary_to_ptm(UnitaryGateRep(H, ("Q0",)))
        result = _ptm_to_unitary(ptm_rep)
        assert isinstance(result, UnitaryGateRep)
        assert np.allclose(result.unitary, H) or np.allclose(
            result.unitary, -H
        )

    def test_fails_for_non_unitary_channel(self):
        ptm_rep = _kraus_to_ptm(
            KrausGateRep(_depolarizing_kraus_ops(0.3), ("Q0",))
        )
        with pytest.raises(RepConstructionError):
            _ptm_to_unitary(ptm_rep)

    def test_unitarity_check_abstol_none_skips_check_but_not_structural_one(
        self,
    ):
        """`unitarity_check_abstol=None` accepts a single non-unitary Choi
        term, but the "exactly one term" structural check still applies."""
        ptm_rep = _kraus_to_ptm(
            KrausGateRep(_amplitude_damping_kraus_ops(0.4), ("Q0",))
        )
        with pytest.raises(RepConstructionError):
            _ptm_to_unitary(ptm_rep, unitarity_check_abstol=None)

        non_unitary = np.array([[1, 0], [0, 0]], dtype=complex)
        ptm_rep = _unitary_to_ptm(UnitaryGateRep(non_unitary, ("Q0",)))
        result = _ptm_to_unitary(ptm_rep, unitarity_check_abstol=None)
        assert isinstance(result, UnitaryGateRep)


class TestKrausToUnitary:
    def test_succeeds_for_single_unitary_operator(self):
        H = STANDARD_GATE_UNITARIES["H"]
        rep = KrausGateRep([(H, 1.0)], ("Q0",))
        result = _kraus_to_unitary(rep)
        assert isinstance(result, UnitaryGateRep)
        assert np.array_equal(result.unitary, H)

    def test_fails_for_multiple_operators(self):
        rep = KrausGateRep(_depolarizing_kraus_ops(0.3), ("Q0",))
        with pytest.raises(RepConstructionError):
            _kraus_to_unitary(rep)


class TestPTMQSimSuperoperatorRoundTrip:
    @pytest.mark.parametrize("case", ["1q", "2q", "non-unital"])
    def test_round_trip(self, case):
        if case == "1q":
            X = STANDARD_GATE_UNITARIES["X"]
            ptm_rep = _unitary_to_ptm(UnitaryGateRep(X, ("Q0",)))
        elif case == "2q":
            CX = STANDARD_GATE_UNITARIES["CX"]
            ptm_rep = _unitary_to_ptm(UnitaryGateRep(CX, ("Q0", "Q1")))
        else:
            ptm_rep = _kraus_to_ptm(
                KrausGateRep(_amplitude_damping_kraus_ops(0.4), ("Q0",))
            )
        qsim_rep = _ptm_to_qsim_superoperator(ptm_rep)
        assert isinstance(qsim_rep, QSimSuperopGateRep)
        back = _qsim_superoperator_to_ptm(qsim_rep)
        assert np.allclose(back.ptm, ptm_rep.ptm)

    @pytest.mark.skipif(NO_PYGSTI, reason="pyGSTi not installed")
    def test_matches_pygsti_change_basis(self):
        from pygsti.baseobjs import ExplicitBasis
        from pygsti.tools import basistools as bt

        X = STANDARD_GATE_UNITARIES["X"]
        ptm_rep = _unitary_to_ptm(UnitaryGateRep(X, ("Q0",)))
        mine = _ptm_to_qsim_superoperator(ptm_rep).superop

        qsim1 = list(_qsim_basis(1))
        qbasis_obj = ExplicitBasis(
            qsim1, ["a", "b", "c", "d"], name="qsim1", longname="q"
        )
        theirs = bt.change_basis(
            ptm_rep.ptm, pygsti.BuiltinBasis("pp", 4), qbasis_obj
        )
        assert np.allclose(mine, theirs, atol=1e-8)

    def test_qsim_superoperator_fails_for_3_qubits(self):
        # 3-qubit "unitary" (bare identity-shaped array is enough since the
        # qsim-basis lookup itself is what should reject this).
        rep = PTMGateRep(np.eye(64), ("Q0", "Q1", "Q2"))
        with pytest.raises(RepConstructionError):
            _ptm_to_qsim_superoperator(rep)


class TestShortestPath:
    def test_multi_hop_prefers_shortest(self):
        # Kraus -> QSimSuperoperator has no direct edge; must go via PTM.
        path = _shortest_path(KrausGateRep, QSimSuperopGateRep)
        assert path == [KrausGateRep, PTMGateRep, QSimSuperopGateRep]


class TestZBasisProjectionToZBasisPrePost:
    def test_always_succeeds_with_identity_ops(self):
        rep = ZBasisProjectionInstrumentRep(0, True, ("Q0",))
        pp = _zbasis_projection_to_zbasis_pre_post(rep)
        assert isinstance(pp, ZBasisPrePostInstrumentRep)
        assert pp.reset == 0
        assert pp.include_outcome is True
        assert _is_identity_gaterep(pp.pre_op)
        assert _is_identity_gaterep(pp.post_op)


class TestZBasisPrePostToZBasisProjection:
    def test_succeeds_for_identity_pre_post(self):
        identity = UnitaryGateRep(np.eye(2), ("Q0",))
        rep = ZBasisPrePostInstrumentRep(1, False, identity, identity, ("Q0",))
        zp = _zbasis_pre_post_to_zbasis_projection(rep)
        assert isinstance(zp, ZBasisProjectionInstrumentRep)
        assert zp.reset == 1
        assert zp.include_outcome is False

    def test_fails_for_non_identity_pre_op(self):
        X = STANDARD_GATE_UNITARIES["X"]
        identity = UnitaryGateRep(np.eye(2), ("Q0",))
        noisy = UnitaryGateRep(X, ("Q0",))
        rep = ZBasisPrePostInstrumentRep(None, True, noisy, identity, ("Q0",))
        with pytest.raises(RepConstructionError):
            _zbasis_pre_post_to_zbasis_projection(rep)


class TestZBasisProjectionOutcomeOperationDictRoundTrip:
    @pytest.mark.parametrize("reset", [None, 0, 1])
    def test_round_trip(self, reset):
        zp = ZBasisProjectionInstrumentRep(reset, True, ("Q0",))
        od = _zbasis_projection_to_outcome_operation_dict(zp)
        assert isinstance(od, OutcomeOperationDictInstrumentRep)
        assert set(od.outcome_ops.keys()) == {0, 1}
        back = _outcome_operation_dict_to_zbasis_projection(od)
        assert back.reset == reset
        assert back.include_outcome is True

    def test_outcome_operators_match_simulated_behavior(self):
        """Physics sanity check: the constructed outcome_ops must produce
        the same measurement statistics/post-measurement state as
        directly applying the equivalent ZBasisProjectionInstrumentRep."""
        from loqs.backends import NumpyStatevectorQuantumState as SVState

        for reset in (None, 0, 1):
            zp = ZBasisProjectionInstrumentRep(reset, True, ("Q0",))
            od = _zbasis_projection_to_outcome_operation_dict(zp)
            for seed in range(10):
                s1 = SVState(
                    np.array([1.0, 1.0]) / np.sqrt(2), ["Q0"], seed=seed
                )
                out1 = s1.apply_reps_inplace([zp])
                s2 = SVState(
                    np.array([1.0, 1.0]) / np.sqrt(2), ["Q0"], seed=seed
                )
                out2 = s2.apply_reps_inplace([od])
                assert out1["Q0"] == out2["Q0"]
                assert np.allclose(s1.state, s2.state)

    def test_fails_for_more_than_one_qubit(self):
        zp = ZBasisProjectionInstrumentRep(None, True, ("Q0", "Q1"))
        with pytest.raises(RepConstructionError):
            _zbasis_projection_to_outcome_operation_dict(zp)

    def test_fails_for_wrong_outcome_keys(self):
        identity = UnitaryGateRep(np.eye(2), ("Q0",))
        od = OutcomeOperationDictInstrumentRep(
            {0: identity, 2: identity}, True, ("Q0",)
        )
        with pytest.raises(RepConstructionError):
            _outcome_operation_dict_to_zbasis_projection(od)

    def test_fails_for_non_projector_outcome_operator(self):
        X = STANDARD_GATE_UNITARIES["X"]
        od = OutcomeOperationDictInstrumentRep(
            {
                0: UnitaryGateRep(np.eye(2), ("Q0",)),
                1: UnitaryGateRep(X, ("Q0",)),
            },
            True,
            ("Q0",),
        )
        with pytest.raises(RepConstructionError):
            _outcome_operation_dict_to_zbasis_projection(od)

    def test_reverse_direction_also_fails_for_more_than_one_qubit(self):
        identity = UnitaryGateRep(np.eye(4), ("Q0", "Q1"))
        od = OutcomeOperationDictInstrumentRep(
            {(0, 0): identity, (1, 1): identity}, True, ("Q0", "Q1")
        )
        with pytest.raises(RepConstructionError):
            _outcome_operation_dict_to_zbasis_projection(od)

    def test_fails_for_non_unitarygaterep_outcome_operator(self):
        od = OutcomeOperationDictInstrumentRep(
            {
                0: PTMGateRep(np.eye(4), ("Q0",)),
                1: UnitaryGateRep(np.eye(2), ("Q0",)),
            },
            True,
            ("Q0",),
        )
        with pytest.raises(RepConstructionError):
            _outcome_operation_dict_to_zbasis_projection(od)

    def test_joint_outcome_qubits_does_not_bypass_one_qubit_restriction(self):
        """A joint (`len(outcome_qubits) == 1`) instrument on more than one
        physical qubit -- e.g. a 2Q parity check -- must still be rejected
        by both conversion directions; `outcome_qubits` only affects
        classical-label bookkeeping, not this conversion's qubit-count
        scope."""
        even = UnitaryGateRep(np.diag([1.0, 0, 0, 1.0]), ("Q0", "Q1"))
        odd = UnitaryGateRep(np.diag([0, 1.0, 1.0, 0]), ("Q0", "Q1"))
        od = OutcomeOperationDictInstrumentRep(
            {"even": even, "odd": odd},
            True,
            ("Q0", "Q1"),
            outcome_qubits="synd",
        )
        assert od.outcome_qubits == ("synd",)
        with pytest.raises(RepConstructionError):
            _outcome_operation_dict_to_zbasis_projection(od)

    def test_fails_for_swapped_targets(self):
        """Both `outcome_ops` entries are individually clean permutation
        projectors, but `{0: |1><0|, 1: |0><1|}` ("swap the outcomes") does
        not correspond to any `ZBasisProjectionInstrumentRep` (reset is
        either `None` -> identity targets, or a fixed value -> both
        outcomes map to the *same* target -- never a swap)."""
        swapped_0 = np.array([[0, 0], [1, 0]], dtype=complex)  # |1><0|
        swapped_1 = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|
        od = OutcomeOperationDictInstrumentRep(
            {
                0: UnitaryGateRep(swapped_0, ("Q0",)),
                1: UnitaryGateRep(swapped_1, ("Q0",)),
            },
            True,
            ("Q0",),
        )
        with pytest.raises(RepConstructionError):
            _outcome_operation_dict_to_zbasis_projection(od)


class TestZBasisProjectionStimCircuitRoundTrip:
    @pytest.mark.parametrize(
        "reset,include_outcome,expected",
        [
            (None, True, "M 0"),
            (0, True, "MR 0"),
            (0, False, "R 0"),
            (1, True, "MR 0\nX 0"),
            (1, False, "R 0\nX 0"),
        ],
    )
    def test_forward_mapping(self, reset, include_outcome, expected):
        rep = ZBasisProjectionInstrumentRep(reset, include_outcome, ("Q0",))
        stim_rep = _zbasis_projection_to_stim_circuit(rep)
        assert isinstance(stim_rep, StimCircuitInstrumentRep)
        assert stim_rep.circuit_str == expected

    def test_none_false_is_unrepresentable(self):
        rep = ZBasisProjectionInstrumentRep(None, False, ("Q0",))
        with pytest.raises(RepConstructionError):
            _zbasis_projection_to_stim_circuit(rep)

    @pytest.mark.parametrize("reset", [None, 0, 1])
    @pytest.mark.parametrize("include_outcome", [True, False])
    def test_round_trip(self, reset, include_outcome):
        if reset is None and not include_outcome:
            pytest.skip("(None, False) is intentionally unrepresentable")
        rep = ZBasisProjectionInstrumentRep(reset, include_outcome, ("Q0",))
        stim_rep = _zbasis_projection_to_stim_circuit(rep)
        back = _stim_circuit_to_zbasis_projection(stim_rep)
        assert back.reset == reset
        assert back.include_outcome == include_outcome

    def test_multi_qubit_forward_mapping(self):
        rep = ZBasisProjectionInstrumentRep(1, True, ("Q0", "Q1"))
        stim_rep = _zbasis_projection_to_stim_circuit(rep)
        assert stim_rep.circuit_str == "MR 0 1\nX 0 1"
        back = _stim_circuit_to_zbasis_projection(stim_rep)
        assert (back.reset, back.include_outcome) == (1, True)

    def test_rejects_x_basis_measurement(self):
        rep = StimCircuitInstrumentRep("MX 0", ("Q0",))
        with pytest.raises(RepConstructionError):
            _stim_circuit_to_zbasis_projection(rep)

    def test_rejects_unrelated_multiline_circuit(self):
        rep = StimCircuitInstrumentRep("M 0\nM 0", ("Q0",))
        with pytest.raises(RepConstructionError):
            _stim_circuit_to_zbasis_projection(rep)

    def test_rejects_mismatched_qubit_targets(self):
        rep = StimCircuitInstrumentRep("MR 0\nX 1", ("Q0",))
        with pytest.raises(RepConstructionError):
            _stim_circuit_to_zbasis_projection(rep)

    def test_rejects_circuit_with_more_than_two_lines(self):
        rep = StimCircuitInstrumentRep("MR 0\nX 0\nX 0", ("Q0",))
        with pytest.raises(RepConstructionError):
            _stim_circuit_to_zbasis_projection(rep)

    def test_rejects_empty_circuit(self):
        rep = StimCircuitInstrumentRep("", ("Q0",))
        with pytest.raises(RepConstructionError):
            _stim_circuit_to_zbasis_projection(rep)


@pytest.mark.skipif(NO_STIM, reason="stim is not installed")
class TestUnitaryStimCircuitRoundTrip:
    @pytest.mark.parametrize("name", ["H", "S", "S_DAG", "CX", "CZ"])
    def test_clifford_round_trips(self, name):
        U = STANDARD_GATE_UNITARIES[name]
        qubits = ("Q0", "Q1") if U.shape[0] == 4 else ("Q0",)
        rep = UnitaryGateRep(U, qubits)
        stim_rep = _unitary_to_stim_circuit(rep)
        assert isinstance(stim_rep, StimCircuitGateRep)
        back = _stim_circuit_to_unitary(stim_rep)
        # Up to global phase.
        overlap = abs(np.vdot(back.unitary, U)) / U.shape[0]
        assert np.isclose(overlap, 1.0, atol=1e-8)

    def test_rejects_non_clifford_unitary(self):
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        rep = UnitaryGateRep(T, ("Q0",))
        with pytest.raises(RepConstructionError):
            _unitary_to_stim_circuit(rep)

    def test_rejects_noise_circuit(self):
        rep = StimCircuitGateRep("X_ERROR(0.1) 0", ("Q0",))
        with pytest.raises(RepConstructionError):
            _stim_circuit_to_unitary(rep)

    def test_matches_simulation_across_backends(self):
        """Physics ground truth: applying a UnitaryGateRep via the
        NumpyStatevector backend must agree with applying its
        STIM-converted equivalent via the STIM backend, for every
        computational basis input."""
        from loqs.backends import NumpyStatevectorQuantumState as SVState
        from loqs.backends import STIMQuantumState as StimState

        def _bits(basis_idx, n):
            return [(basis_idx >> (n - 1 - i)) & 1 for i in range(n)]

        for name in ["H", "X", "CX", "CZ"]:
            U = STANDARD_GATE_UNITARIES[name]
            n = 2 if U.shape[0] == 4 else 1
            qubits = ("Q0", "Q1") if n == 2 else ("Q0",)
            urep = UnitaryGateRep(U, qubits)
            stim_rep = _unitary_to_stim_circuit(urep)

            for basis_idx in range(2**n):
                init = np.zeros(2**n)
                init[basis_idx] = 1.0
                s1 = SVState(init.reshape((2,) * n), qubits)
                s1.apply_reps_inplace([urep])
                final1 = s1.state.reshape(-1)

                s2 = StimState(_bits(basis_idx, n), qubits)
                s2.apply_reps_inplace([stim_rep])
                final2 = s2.state.state_vector(endian="big")

                assert np.allclose(np.abs(final1), np.abs(final2), atol=1e-6)


@pytest.mark.skipif(not NO_STIM, reason="stim IS installed")
class TestUnitaryStimCircuitEdgeAbsentWithoutStim:
    def test_edge_not_registered(self):
        from loqs.backends.reps.conversion import _CONVERTERS

        assert (UnitaryGateRep, StimCircuitGateRep) not in _CONVERTERS
        assert (StimCircuitGateRep, UnitaryGateRep) not in _CONVERTERS


class TestConvert:
    def test_passthrough_for_instance_matching_one_of_a_target_list(self):
        from loqs.backends.reps.conversion import convert

        rep = PTMGateRep(np.eye(4), ("Q0",))
        assert convert(rep, [UnitaryGateRep, PTMGateRep]) is rep

    def test_multi_hop_from_instance(self):
        from loqs.backends.reps.conversion import convert

        H = STANDARD_GATE_UNITARIES["H"]
        rep = UnitaryGateRep(H, ("Q0",))
        result = convert(rep, QSimSuperopGateRep)
        assert isinstance(result, QSimSuperopGateRep)

    def test_multi_target_hop_picks_shortest_path_regardless_of_list_order(
        self,
    ):
        """Unlike the raw-payload direct-match case (which uses list
        order as a priority), hopping from an already-typed instance
        picks whichever target has the *shortest* path -- a later,
        longer-path candidate must not override an already-found
        shorter one."""
        from loqs.backends.reps.conversion import convert

        H = STANDARD_GATE_UNITARIES["H"]
        rep = UnitaryGateRep(H, ("Q0",))
        # PTMGateRep is 1 hop away; QSimSuperopGateRep is 2 hops away
        # (via PTM) -- listed in that (shorter-first) order on purpose.
        result = convert(rep, [PTMGateRep, QSimSuperopGateRep])
        assert isinstance(result, PTMGateRep)

    def test_raises_when_instance_has_no_path_to_target(self):
        from loqs.backends.reps.conversion import convert

        rep = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(RepConstructionError):
            convert(rep, ZBasisProjectionInstrumentRep)

    def test_raw_payload_ambiguous_starting_class_raises(self):
        from loqs.backends.reps.conversion import convert

        # (4, 4) with 1 qubit is ambiguous between PTM and QSimSuperoperator
        # -- since KrausGateRep isn't a direct target match either, this
        # must raise rather than silently guess a starting point.
        with pytest.raises(RepConstructionError):
            convert(np.eye(4), KrausGateRep, ("Q0",))

    def test_kwargs_forwarded_to_hop_converters(self):
        """`unitarity_check_abstol=None` reaches `_kraus_to_unitary` even
        when hopping from an already-constructed `KrausGateRep` instance,
        not just when constructing directly from a raw payload."""
        from loqs.backends.reps.conversion import convert

        non_unitary = np.eye(2) * 0.5
        rep = KrausGateRep([(non_unitary, 1.0)], ("Q0",), tp_check_abstol=None)

        with pytest.raises(RepConstructionError):
            convert(rep, UnitaryGateRep)

        result = convert(rep, UnitaryGateRep, unitarity_check_abstol=None)
        assert isinstance(result, UnitaryGateRep)
        assert np.array_equal(result.unitary, non_unitary)

    def test_raw_str_direct_match_to_stim_instrument_rep(self):
        from loqs.backends.reps.conversion import convert

        result = convert(
            "M 0",
            [StimCircuitInstrumentRep, ZBasisProjectionInstrumentRep],
            ("Q0",),
        )
        assert isinstance(result, StimCircuitInstrumentRep)
