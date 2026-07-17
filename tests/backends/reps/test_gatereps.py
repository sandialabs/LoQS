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
        ultimately aren't selected (e.g. by `upgrade_gate_rep`)."""
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
