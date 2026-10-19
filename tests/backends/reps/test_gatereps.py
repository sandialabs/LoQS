"""Tester for loqs.backends.reps.gatereps"""

import numpy as np
import pytest

from loqs.backends.reps import (
    GateRep,
    KrausGateRep,
    PTMGateRep,
    ProbabilisticStimGateRep,
    QSimSuperopGateRep,
    RepConstructionError,
    StimCircuitGateRep,
    UnitaryGateRep,
)

# TP-preserving (sum_i K_i^dagger K_i = I) single-qubit Kraus operators.
_GAMMA = 0.1
_TP_K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - _GAMMA)]])
_TP_K1 = np.array([[0.0, np.sqrt(_GAMMA)], [0.0, 0.0]])

# Deliberately non-TP: a single, non-unitary, non-normalized operator.
_NON_TP_K = np.eye(2) * 0.5


def _assert_kraus_reps_equal(expected: KrausGateRep, actual: KrausGateRep):
    assert isinstance(actual, KrausGateRep)
    assert expected.qubit_labels == actual.qubit_labels
    assert len(expected.kraus_operators) == len(actual.kraus_operators)
    for (ek, ep), (ak, ap) in zip(
        expected.kraus_operators, actual.kraus_operators
    ):
        assert np.allclose(ek, ak)
        if ep is None or ap is None:
            assert ep is None and ap is None
        else:
            assert np.allclose(ep, ap)


@pytest.mark.parametrize(
    "make,attr,expected",
    [
        (lambda: UnitaryGateRep(np.eye(2), ("Q0",)), "unitary", np.eye(2)),
        (lambda: PTMGateRep(np.eye(4), ("Q0",)), "ptm", np.eye(4)),
        (
            lambda: QSimSuperopGateRep(np.eye(4), ("Q0",)),
            "superop",
            np.eye(4),
        ),
        (
            lambda: StimCircuitGateRep("X 0", ("Q0",)),
            "circuit_str",
            "X 0",
        ),
        (
            lambda: ProbabilisticStimGateRep(
                [["X 0", 0.5], ["Y 0", 0.5]], ("Q0",)  # type: ignore[list-item]
            ),
            "operations",
            (("X 0", 0.5), ("Y 0", 0.5)),
        ),
    ],
    ids=["unitary", "ptm", "qsim", "stim", "probabilistic-stim"],
)
def test_constructs_instance(make, attr, expected):
    rep = make()
    assert isinstance(rep, GateRep)
    assert rep.qubit_labels == ("Q0",)
    payload = getattr(rep, attr)
    if isinstance(expected, np.ndarray):
        assert np.array_equal(payload, expected)
    else:
        assert payload == expected


class TestUnitaryGateRep:
    def test_rejects_wrong_shape_for_known_qubits(self):
        with pytest.raises(RepConstructionError):
            UnitaryGateRep(np.eye(4), ("Q0",))

    def test_accepts_ambiguous_shape_without_qubits_attached(self):
        # (4, 4) is a power of both 2 and 4 -- without qubits attached,
        # only the weaker "square, power-of-2" check applies.
        rep = UnitaryGateRep(np.eye(4))
        assert rep.unitary.shape == (4, 4)


class TestStimCircuitGateRep:
    def test_rejects_non_str(self):
        with pytest.raises(RepConstructionError):
            StimCircuitGateRep(42, ("Q0",))


class TestProbabilisticStimGateRep:
    def test_rejects_malformed_sequences(self):
        with pytest.raises(RepConstructionError):
            # element is not a (str, prob) tuple/list
            ProbabilisticStimGateRep(("X 0",), ("Q0",))


class TestKrausGateRep:
    def test_constructs_valid_tp_kraus_no_warning(self, recwarn):
        gr = ((_TP_K0, None), (_TP_K1, None))
        rep = KrausGateRep(gr, ("Q0",))
        assert isinstance(rep, KrausGateRep)
        assert len(recwarn) == 0

    def test_rejects_malformed_sequences(self):
        with pytest.raises(RepConstructionError):
            # element is not an (op, prob) tuple/list at all
            KrausGateRep((_TP_K0,), ("Q0",))

    def test_non_tp_kraus_warns(self):
        with pytest.warns(UserWarning):
            rep = KrausGateRep(((_NON_TP_K, None),), ("Q0",))
        assert isinstance(rep, KrausGateRep)  # still constructed, just warned


class TestKrausGateRepFromPauliStochastic:
    def test_2q_pauli_stochastic_matches_diagonal_ptm(self):
        rng = np.random.default_rng(0)
        non_i_rates = rng.random(15) * 0.05
        rates = [1 - sum(non_i_rates)] + list(non_i_rates)
        rep = KrausGateRep.from_pauli_stochastic(rates, ["Q0", "Q1"])

        from loqs.backends.reps.conversion import _kraus_to_ptm

        ptm = _kraus_to_ptm(rep).ptm
        assert np.allclose(ptm, np.diag(np.diag(ptm)), atol=1e-10)

    def test_negligible_terms_are_omitted(self):
        rates = [1.0, 0.0, 0.0, 0.0]
        rep = KrausGateRep.from_pauli_stochastic(rates, ["Q0"])
        assert len(rep.kraus_operators) == 1


class TestKrausGateRepFromDepolarizing:
    def test_depolarizing_ptm_is_diagonal(self):
        from loqs.backends.reps.conversion import _kraus_to_ptm

        p = 0.1
        rep = KrausGateRep.from_depolarizing(p, ["Q0"])
        ptm = _kraus_to_ptm(rep).ptm
        expected = np.diag([1, 1 - p, 1 - p, 1 - p])
        assert np.allclose(ptm, expected, atol=1e-10)


class TestKrausGateRepFromAmplitudeDamping:
    def test_two_kraus_operators(self):
        rep = KrausGateRep.from_amplitude_damping(0.4, "Q0")
        assert len(rep.kraus_operators) == 2
        assert rep.qubit_labels == ("Q0",)

    def test_action_matches_hand_built_channel(self):
        gamma = 0.4
        rep = KrausGateRep.from_amplitude_damping(gamma, "Q0")
        a0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        a1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        (k0, _), (k1, _) = rep.kraus_operators
        assert np.allclose(k0, a0)
        assert np.allclose(k1, a1)


class TestKrausGateRepDedup:
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
        _assert_kraus_reps_equal(expected, rep.dedup())

    def test_dedup_raises_for_non_unital_operators(self):
        with pytest.raises(ValueError):
            KrausGateRep([(np.eye(2), None)], [0]).dedup()


class TestKrausGateRepCompose:
    def test_compose_with_unitary_gaterep(self):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        rep1 = KrausGateRep([(Z, 1.0)], [0])
        rep2 = UnitaryGateRep(X, [0])

        # Z applied first, then X: X @ Z
        expected = KrausGateRep([(X @ Z, 1.0)], [0])
        _assert_kraus_reps_equal(expected, rep1.compose(rep2))

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
        _assert_kraus_reps_equal(expected, result)

    def test_compose_rejects_non_gaterep_argument(self):
        rep1 = KrausGateRep([(np.eye(2), 1.0)], [0])
        with pytest.raises(TypeError):
            rep1.compose(PTMGateRep(np.eye(4), [0]))
