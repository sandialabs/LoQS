"""Tester for loqs.backends.reps.gatereps"""

import numpy as np
import pytest

from loqs.backends.reps import (
    GateRep,
    KrausGateRep,
    PTMGateRep,
    ProbabilisticStimGateRep,
    QSimSuperoperatorGateRep,
    RepConstructionError,
    StimCircuitGateRep,
    UnitaryGateRep,
)

# TP-preserving (sum_i K_i K_i^dagger = I) single-qubit Kraus operators.
_GAMMA = 0.1
_TP_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_TP_K1 = np.array([[0.0, 0.0], [np.sqrt(_GAMMA), 0.0]])

# Deliberately non-TP: a single, non-unitary, non-normalized operator.
_NON_TP_K = np.eye(2) * 0.5


class TestUnitaryGateRep:
    def test_matches_ndarray(self):
        assert UnitaryGateRep.matches(np.eye(2)) is True

    def test_does_not_match_non_ndarray(self):
        assert UnitaryGateRep.matches("not an array") is False

    def test_from_raw_constructs_instance(self):
        rep = UnitaryGateRep.from_raw(np.eye(2), ("Q0",))
        assert isinstance(rep, UnitaryGateRep)
        assert isinstance(rep, GateRep)
        assert np.array_equal(rep.unitary, np.eye(2))
        assert rep.qubits == ("Q0",)

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            UnitaryGateRep.from_raw("not an array", ("Q0",))


class TestPTMGateRep:
    def test_matches_ndarray(self):
        assert PTMGateRep.matches(np.eye(4)) is True

    def test_from_raw_constructs_instance(self):
        rep = PTMGateRep.from_raw(np.eye(4), ("Q0",))
        assert isinstance(rep, PTMGateRep)
        assert np.array_equal(rep.ptm, np.eye(4))

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            PTMGateRep.from_raw(None, ("Q0",))


class TestQSimSuperoperatorGateRep:
    def test_matches_ndarray(self):
        assert QSimSuperoperatorGateRep.matches(np.eye(4)) is True

    def test_from_raw_constructs_instance(self):
        rep = QSimSuperoperatorGateRep.from_raw(np.eye(4), ("Q0",))
        assert isinstance(rep, QSimSuperoperatorGateRep)
        assert np.array_equal(rep.superop, np.eye(4))

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            QSimSuperoperatorGateRep.from_raw(None, ("Q0",))


class TestStimCircuitGateRep:
    def test_matches_str(self):
        assert StimCircuitGateRep.matches("X 0") is True

    def test_does_not_match_non_str(self):
        assert StimCircuitGateRep.matches(np.eye(2)) is False

    def test_from_raw_constructs_instance(self):
        rep = StimCircuitGateRep.from_raw("X 0", ("Q0",))
        assert isinstance(rep, StimCircuitGateRep)
        assert rep.circuit_str == "X 0"

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            StimCircuitGateRep.from_raw(42, ("Q0",))


class TestProbabilisticStimGateRep:
    def test_matches_valid_sequence(self):
        gr = (("X 0", 0.5), ("Y 0", 0.5))
        assert ProbabilisticStimGateRep.matches(gr) is True

    def test_accepts_int_probability(self):
        assert ProbabilisticStimGateRep.matches((("X 0", 1),)) is True

    def test_empty_sequence_does_not_match(self):
        assert ProbabilisticStimGateRep.matches(()) is False

    def test_does_not_match_bare_string(self):
        assert ProbabilisticStimGateRep.matches("X 0") is False

    @pytest.mark.parametrize(
        "malformed",
        [
            ("X 0",),  # element is not a (str, prob) tuple/list
            (("X 0", 0.5, "extra"),),  # wrong length
            ((0, 0.5),),  # first entry not a str
            (("X 0", "not a number"),),  # second entry not float/int
        ],
    )
    def test_malformed_sequences_do_not_match(self, malformed):
        assert ProbabilisticStimGateRep.matches(malformed) is False

    def test_from_raw_constructs_instance_with_immutable_operations(self):
        rep = ProbabilisticStimGateRep.from_raw(
            [["X 0", 0.5], ["Y 0", 0.5]], ("Q0",)
        )
        assert isinstance(rep, ProbabilisticStimGateRep)
        assert rep.operations == (("X 0", 0.5), ("Y 0", 0.5))

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            ProbabilisticStimGateRep.from_raw(("X 0",), ("Q0",))


class TestKrausGateRep:
    def test_matches_valid_tp_kraus(self):
        gr = ((_TP_K0, None), (_TP_K1, None))
        assert KrausGateRep.matches(gr) is True

    def test_matches_with_precomputed_probabilities(self):
        gr = ((_TP_K0, 0.95), (_TP_K1, 0.05))
        assert KrausGateRep.matches(gr) is True

    def test_empty_sequence_does_not_match(self):
        assert KrausGateRep.matches(()) is False

    def test_does_not_match_bare_string(self):
        assert KrausGateRep.matches("not a sequence of kraus ops") is False

    @pytest.mark.parametrize(
        "malformed",
        [
            (_TP_K0,),  # element is not a (op, prob) tuple/list at all
            ((_TP_K0, 0.5, "extra"),),  # wrong length
            (("not an array", 0.5),),  # first entry not an ndarray
            ((_TP_K0, "not a float"),),  # second entry not float/None
        ],
    )
    def test_malformed_sequences_do_not_match(self, malformed):
        assert KrausGateRep.matches(malformed) is False

    def test_matches_does_not_warn_for_non_tp_kraus(self, recwarn):
        """`matches` is a pure structural check with no side effects --
        the TP-preservation warning only fires from `from_raw`, since
        `matches` may be called speculatively on candidates that
        ultimately aren't selected (e.g. by `convert`)."""
        assert KrausGateRep.matches(((_NON_TP_K, None),)) is True
        assert len(recwarn) == 0

    def test_from_raw_valid_tp_kraus_no_warning(self, recwarn):
        gr = ((_TP_K0, None), (_TP_K1, None))
        rep = KrausGateRep.from_raw(gr, ("Q0",))
        assert isinstance(rep, KrausGateRep)
        assert len(recwarn) == 0

    def test_from_raw_non_tp_kraus_warns(self):
        with pytest.warns(UserWarning, match="do not constitute a TP channel"):
            rep = KrausGateRep.from_raw(((_NON_TP_K, None),), ("Q0",))
        assert isinstance(rep, KrausGateRep)  # still constructed, just warned

    def test_from_raw_tp_check_disabled_by_infinite_tolerance(self, recwarn):
        rep = KrausGateRep.from_raw(
            ((_NON_TP_K, None),), ("Q0",), tp_check_abstol=float("inf")
        )
        assert isinstance(rep, KrausGateRep)
        assert len(recwarn) == 0

    def test_from_raw_rejects_non_matching_payload(self):
        with pytest.raises(RepConstructionError):
            KrausGateRep.from_raw((), ("Q0",))

    def test_kraus_operators_stored_as_immutable_tuple_of_tuples(self):
        rep = KrausGateRep([[_TP_K0, None], [_TP_K1, None]], ("Q0",))
        assert isinstance(rep.kraus_operators, tuple)
        assert all(isinstance(k, tuple) for k in rep.kraus_operators)


class TestKrausGateRepFromPauliStochastic:
    def test_depolarizing_via_pauli_stochastic_1q(self):
        p = 0.1
        rep = KrausGateRep.from_pauli_stochastic(
            [1 - 3 * p / 4, p / 4, p / 4, p / 4], ["Q0"]
        )
        from loqs.backends.reps.conversion import _kraus_to_ptm

        ptm = _kraus_to_ptm(rep).ptm
        expected = np.diag([1, 1 - p, 1 - p, 1 - p])
        assert np.allclose(ptm, expected, atol=1e-10)

    def test_2q_pauli_stochastic_matches_diagonal_ptm(self):
        rng = np.random.default_rng(0)
        non_i_rates = rng.random(15) * 0.05
        rates = [1 - sum(non_i_rates)] + list(non_i_rates)
        rep = KrausGateRep.from_pauli_stochastic(rates, ["Q0", "Q1"])

        from loqs.backends.reps.conversion import _kraus_to_ptm

        ptm = _kraus_to_ptm(rep).ptm
        assert np.allclose(ptm, np.diag(np.diag(ptm)), atol=1e-10)
        assert np.allclose(np.diag(ptm).real, rep_diag_from_rates(rates), atol=1e-8)

    def test_negligible_terms_are_omitted(self):
        rates = [1.0, 0.0, 0.0, 0.0]
        rep = KrausGateRep.from_pauli_stochastic(rates, ["Q0"])
        assert len(rep.kraus_operators) == 1


def rep_diag_from_rates(rates):
    """Cross-check helper: PTM diagonal (Pauli eigenvalues) for a Pauli-
    stochastic channel, computed directly from the Kraus/PTM machinery
    rather than the deleted Walsh-Hadamard-transform shortcut."""
    from loqs.backends.reps.conversion import _kraus_to_ptm

    rep = KrausGateRep.from_pauli_stochastic(rates, ["Q0", "Q1"])
    return np.diag(_kraus_to_ptm(rep).ptm).real


class TestKrausGateRepFromDepolarizing:
    def test_matches_from_pauli_stochastic(self):
        p = 0.1
        via_depolarizing = KrausGateRep.from_depolarizing(p, ["Q0"])
        via_pauli_stochastic = KrausGateRep.from_pauli_stochastic(
            [1 - 3 * p / 4, p / 4, p / 4, p / 4], ["Q0"]
        )
        assert len(via_depolarizing.kraus_operators) == len(
            via_pauli_stochastic.kraus_operators
        )
        for (k1, p1), (k2, p2) in zip(
            via_depolarizing.kraus_operators, via_pauli_stochastic.kraus_operators
        ):
            assert np.allclose(k1, k2)
            assert np.isclose(p1, p2)


class TestKrausGateRepFromAmplitudeDamping:
    def test_two_kraus_operators(self):
        rep = KrausGateRep.from_amplitude_damping(0.4, "Q0")
        assert len(rep.kraus_operators) == 2
        assert rep.qubits == ("Q0",)

    def test_action_matches_hand_built_channel(self):
        gamma = 0.4
        rep = KrausGateRep.from_amplitude_damping(gamma, "Q0")
        a0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        a1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        (k0, _), (k1, _) = rep.kraus_operators
        assert np.allclose(k0, a0)
        assert np.allclose(k1, a1)


class TestKrausGateRepDedup:
    def _assert_kraus_reps_equal(self, expected: KrausGateRep, actual: KrausGateRep):
        assert isinstance(actual, KrausGateRep)
        assert expected.qubits == actual.qubits
        assert len(expected.kraus_operators) == len(actual.kraus_operators)
        for (ek, ep), (ak, ap) in zip(
            expected.kraus_operators, actual.kraus_operators
        ):
            assert np.allclose(ek, ak)
            assert np.allclose(ep, ap)

    def test_dedup_simple_duplicate(self):
        rep = KrausGateRep(
            [(np.sqrt(0.6) * np.eye(2), 0.6), (np.sqrt(0.4) * np.eye(2), 0.4)], [0]
        )
        expected = KrausGateRep([(np.eye(2), 1.0)], [0])
        self._assert_kraus_reps_equal(expected, rep.dedup())

    def test_dedup_more_complicated_mix(self):
        rep = KrausGateRep(
            [
                (np.sqrt(0.4) * np.eye(2), 0.4),
                (np.sqrt(0.3) * np.eye(2), 0.3),
                (np.sqrt(0.1) * np.eye(2), 0.1),
                (np.sqrt(0.1) * np.array([[0, 1], [1, 0]]), 0.1),
                (np.sqrt(0.05) * np.array([[0, 1], [1, 0]]), 0.05),
                (np.sqrt(0.05) * np.array([[1, 0], [0, -1]]), 0.05),
            ],
            [0],
        )
        expected = KrausGateRep(
            [
                (np.sqrt(0.8) * np.eye(2), 0.8),
                (np.sqrt(0.15) * np.array([[0, 1], [1, 0]]), 0.15),
                (np.sqrt(0.05) * np.array([[1, 0], [0, -1]]), 0.05),
            ],
            [0],
        )
        self._assert_kraus_reps_equal(expected, rep.dedup())

    def test_dedup_raises_for_non_unital_operators(self):
        with pytest.raises(ValueError):
            KrausGateRep([(np.eye(2), None)], [0]).dedup()


class TestKrausGateRepCompose:
    def _assert_kraus_reps_equal(self, expected: KrausGateRep, actual: KrausGateRep):
        assert isinstance(actual, KrausGateRep)
        assert expected.qubits == actual.qubits
        assert len(expected.kraus_operators) == len(actual.kraus_operators)
        for (ek, ep), (ak, ap) in zip(
            expected.kraus_operators, actual.kraus_operators
        ):
            assert np.allclose(ek, ak)
            assert np.allclose(ep, ap)

    def test_compose_with_unitary_gaterep(self):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        rep1 = KrausGateRep([(Z, 1.0)], [0])
        rep2 = UnitaryGateRep(X, [0])

        # Z applied first, then X: X @ Z
        expected = KrausGateRep([(X @ Z, 1.0)], [0])
        self._assert_kraus_reps_equal(expected, rep1.compose(rep2))

    def test_compose_two_kraus_reps(self):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        rep1 = KrausGateRep([(Z, 1.0)], [0])
        rep2 = KrausGateRep(
            [(np.sqrt(0.6) * np.eye(2), 0.6), (np.sqrt(0.4) * X, 0.4)], [0]
        )

        # Z o ([I,X]) = [Z, ZX] -> applied Z first, then [I,X]
        expected = KrausGateRep(
            [
                (np.sqrt(0.6) * Z, 0.6),
                (np.sqrt(0.4) * (X @ Z), 0.4),
            ],
            [0],
        )
        self._assert_kraus_reps_equal(expected, rep1.compose(rep2))

    def test_compose_without_dedup(self):
        X = np.array([[0, 1], [1, 0]])
        rep1 = KrausGateRep(
            [(np.sqrt(0.6) * np.eye(2), 0.6), (np.sqrt(0.4) * X, 0.4)], [0]
        )
        rep2 = KrausGateRep(
            [(np.sqrt(0.3) * X, 0.3), (np.sqrt(0.7) * np.eye(2), 0.7)], [0]
        )

        result = rep1.compose(rep2, dedup=False)
        expected = KrausGateRep(
            [
                (np.sqrt(0.6 * 0.3) * X, 0.6 * 0.3),
                (np.sqrt(0.6 * 0.7) * np.eye(2), 0.6 * 0.7),
                (np.sqrt(0.4 * 0.3) * np.eye(2), 0.4 * 0.3),
                (np.sqrt(0.4 * 0.7) * X, 0.4 * 0.7),
            ],
            [0],
        )
        self._assert_kraus_reps_equal(expected, result)

    def test_compose_with_dedup(self):
        X = np.array([[0, 1], [1, 0]])
        rep1 = KrausGateRep(
            [(np.sqrt(0.6) * np.eye(2), 0.6), (np.sqrt(0.4) * X, 0.4)], [0]
        )
        rep2 = KrausGateRep(
            [(np.sqrt(0.3) * X, 0.3), (np.sqrt(0.7) * np.eye(2), 0.7)], [0]
        )

        result = rep1.compose(rep2, dedup=True)
        expected = KrausGateRep(
            [
                (np.sqrt(0.6 * 0.3 + 0.4 * 0.7) * X, 0.6 * 0.3 + 0.4 * 0.7),
                (
                    np.sqrt(0.6 * 0.7 + 0.4 * 0.3) * np.eye(2),
                    0.6 * 0.7 + 0.4 * 0.3,
                ),
            ],
            [0],
        )
        self._assert_kraus_reps_equal(expected, result)

    def test_compose_rejects_non_gaterep_argument(self):
        rep1 = KrausGateRep([(np.eye(2), 1.0)], [0])
        with pytest.raises(TypeError):
            rep1.compose(PTMGateRep(np.eye(4), [0]))
