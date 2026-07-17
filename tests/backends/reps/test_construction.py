"""Tester for loqs.backends.reps.construction"""

import numpy as np
import pytest

from loqs.backends.reps import (
    KrausGateRep,
    ProbabilisticStimGateRep,
    RepConstructionError,
    StimCircuitGateRep,
    StimCircuitInstrumentRep,
    UnitaryGateRep,
    ZBasisProjectionInstrumentRep,
    upgrade_gate_rep,
    upgrade_instrument_rep,
)


class TestUpgradeGateRep:
    def test_passthrough_for_already_constructed_rep(self):
        rep = UnitaryGateRep(np.eye(2), ("Q0",))
        result = upgrade_gate_rep(rep, ("Q0",), [UnitaryGateRep])
        assert result is rep

    def test_passthrough_asserts_type_is_in_allowed(self):
        rep = UnitaryGateRep(np.eye(2), ("Q0",))
        with pytest.raises(AssertionError):
            upgrade_gate_rep(rep, ("Q0",), [StimCircuitGateRep])

    def test_upgrades_raw_str_to_first_matching_class(self):
        result = upgrade_gate_rep(
            "X 0", ("Q0",), [StimCircuitGateRep, ProbabilisticStimGateRep]
        )
        assert isinstance(result, StimCircuitGateRep)
        assert result.circuit_str == "X 0"

    def test_upgrades_raw_sequence_to_kraus(self):
        gr = ((np.eye(2), 1.0),)
        result = upgrade_gate_rep(
            gr, ("Q0",), [StimCircuitGateRep, KrausGateRep]
        )
        assert isinstance(result, KrausGateRep)

    def test_order_determines_which_class_wins_for_ambiguous_payload(self):
        """A bare ndarray structurally matches Unitary/PTM/QSim reps
        equally -- the first class in `allowed` wins."""
        from loqs.backends.reps import PTMGateRep, QSimSuperoperatorGateRep

        result = upgrade_gate_rep(
            np.eye(2), ("Q0",), [PTMGateRep, UnitaryGateRep]
        )
        assert isinstance(result, PTMGateRep)

        result2 = upgrade_gate_rep(
            np.eye(2), ("Q0",), [QSimSuperoperatorGateRep, PTMGateRep]
        )
        assert isinstance(result2, QSimSuperoperatorGateRep)

    def test_no_match_raises_rep_construction_error(self):
        with pytest.raises(RepConstructionError, match="Could not match"):
            upgrade_gate_rep(42, ("Q0",), [StimCircuitGateRep, KrausGateRep])

    def test_kwargs_forwarded_to_from_raw(self):
        # tp_check_abstol=inf should suppress the Kraus TP-check warning
        non_tp = np.eye(2) * 0.5
        result = upgrade_gate_rep(
            ((non_tp, None),),
            ("Q0",),
            [KrausGateRep],
            tp_check_abstol=float("inf"),
        )
        assert isinstance(result, KrausGateRep)


class TestUpgradeInstrumentRep:
    def test_passthrough_for_already_constructed_rep(self):
        rep = ZBasisProjectionInstrumentRep(0, True, ("Q0",))
        result = upgrade_instrument_rep(
            rep, ("Q0",), [ZBasisProjectionInstrumentRep]
        )
        assert result is rep

    def test_passthrough_asserts_type_is_in_allowed(self):
        rep = ZBasisProjectionInstrumentRep(0, True, ("Q0",))
        with pytest.raises(AssertionError):
            upgrade_instrument_rep(
                rep, ("Q0",), [StimCircuitInstrumentRep]
            )

    def test_upgrades_raw_str_to_stim_circuit_instrument_rep(self):
        result = upgrade_instrument_rep(
            "M 0",
            ("Q0",),
            [StimCircuitInstrumentRep, ZBasisProjectionInstrumentRep],
        )
        assert isinstance(result, StimCircuitInstrumentRep)

    def test_upgrades_raw_pair_to_zbasis_projection(self):
        result = upgrade_instrument_rep(
            (0, True),
            ("Q0",),
            [StimCircuitInstrumentRep, ZBasisProjectionInstrumentRep],
        )
        assert isinstance(result, ZBasisProjectionInstrumentRep)

    def test_no_match_raises_rep_construction_error(self):
        with pytest.raises(RepConstructionError, match="Could not match"):
            upgrade_instrument_rep(42, ("Q0",), [StimCircuitInstrumentRep])
